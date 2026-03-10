# Research: Flash Attention 2 + Bidirectional Attention Masks in HuggingFace Transformers

## Executive Summary

**A 4D all-zeros mask passed to a HuggingFace model with `attn_implementation="flash_attention_2"` will NOT silently fall back to causal.** In modern transformers (>=5.x), `create_causal_mask` detects the 4D mask and returns it as-is, bypassing FA2's mask pipeline entirely. However, this means the 4D mask reaches `_flash_attention_forward`, which only handles 2D masks -- so it will either **crash** (when padding is present and `_upad_input` is called on a 4D tensor) or **produce correct bidirectional results** (when no padding exists and `attention_mask=None` is effectively used with `is_causal=False`). The safe approach for FA2 is to NOT pass a 4D mask and instead ensure `is_causal=False` reaches the flash attention kernel directly, which is what both LLaDA and the dllm library do.

## Problem Statement

Our `BidirectionalTransformer` (in `src/diffusion_lm/models/backbone.py`) currently injects an all-zeros 4D float mask `(B, 1, L, L)` to force bidirectional attention. This works perfectly with `attn_implementation="eager"` and `"sdpa"`. The question is whether it works correctly with `attn_implementation="flash_attention_2"` on GPU, which is the production configuration for training at scale.

## Analysis of the HF Transformers Code Path

### Step 1: Model.forward() -> create_causal_mask

When you call `model(input_ids, attention_mask=our_4d_mask)`, the model's `.forward()` method (e.g., `Qwen2Model`, `LlamaModel`, `Qwen3Model`) calls `create_causal_mask()` from `masking_utils.py`.

Inside `create_causal_mask`, the `_preprocess_mask_arguments` function checks:
```python
if isinstance(attention_mask, (torch.Tensor, BlockMask)) and len(attention_mask.shape) == 4:
    return True, attention_mask, ...  # early exit, return 4D mask as-is
```

**Result: Our 4D mask bypasses all causal mask generation and is returned unchanged.** This is correct behavior for "eager" and "sdpa" backends.

### Step 2: The 4D mask reaches the attention layer

For Qwen2/Qwen3/LLaMA, the mask flows:
```
Model.forward() -> decoder_layer() -> self_attn() -> attention_interface()
```

The `attention_interface` is dispatched via `ALL_ATTENTION_FUNCTIONS[config._attn_implementation]`.

### Step 3: flash_attention_forward receives the 4D mask

The `flash_attention_forward` function (in `integrations/flash_attention.py`) passes the mask to `_flash_attention_forward` (in `modeling_flash_attention_utils.py`).

**Critical: `_flash_attention_forward` only handles 2D masks.** It expects `attention_mask` of shape `(batch_size, seq_len)` with 1=real, 0=padding. When a mask is present, it calls `_upad_input(attention_mask)` which does `attention_mask.sum(dim=-1)` and `attention_mask.flatten()` -- operations that will produce wrong shapes on a 4D tensor.

### Step 4: What actually happens

Two scenarios:

**Scenario A: No padding tokens (common in pretraining with packed sequences)**
- `create_causal_mask` with FA2's `flash_attention_mask` function would normally return `None` (no padding -> no mask needed, use `is_causal` flag instead)
- BUT our 4D mask bypasses `flash_attention_mask` entirely via the early exit
- The 4D mask reaches `_flash_attention_forward`
- If `_flash_attention_forward` treats a non-None mask as 2D, it will crash or produce wrong results

**Scenario B: With padding tokens**
- Same flow, but `_upad_input` is definitely called
- `attention_mask.sum(dim=-1)` on a `(B, 1, L, L)` tensor produces `(B, 1, L)` instead of expected `(B,)` -> **crash or silent corruption**

### Transformers Version History

| Version | Mask System | Behavior with 4D + FA2 |
|---------|-------------|------------------------|
| 4.38-4.45 | `_update_causal_mask` method on model | 4D mask early-returns, reaches FA2 layer directly |
| 4.46-5.2 | `_update_causal_mask` + `_prepare_4d_causal_attention_mask_for_flash` | FA2 prep function converts 4D -> 2D or None |
| 5.3+ | `masking_utils.create_causal_mask` + `AttentionMaskInterface` | 4D early exit bypasses `flash_attention_mask` function |

**The PR #39707 (merged Aug 2025)** fixed `is_causal` being ignored in `flash_attention_forward` -- it now respects `is_causal=False` passed via kwargs. This is the correct mechanism for bidirectional attention with FA2.

## How Reference Implementations Handle This

### LLaDA (Official HF Model) -- Custom Model, Does NOT Use HF Attention Pipeline

LLaDA uses a **completely custom model class** (`LLaDAModelLM`) that bypasses HF's attention infrastructure entirely:

```python
# In _scaled_dot_product_attention:
if self.flash_attn_func is not None and attn_mask is None:
    # Call flash_attn directly with causal=False
    r = self.flash_attn_func(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        dropout_p=dropout_p, causal=False  # <-- explicit
    )
else:
    # Fallback to PyTorch SDPA with is_causal=False
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=False  # <-- explicit
    )
```

LLaDA also creates a 4D zero bias `(1, 1, L, L)` via `get_bidirectional_attention_bias()`, but this bias is **only used as `attn_mask` for the SDPA fallback path**. When Flash Attention is active, the bias is NOT passed -- instead, `causal=False` alone controls bidirectionality.

**Key insight: LLaDA passes `attn_mask=None` to flash_attn when FA is enabled.** The zeros bias is skipped entirely.

### dllm Library (Dream Model) -- Hardcoded is_causal=False

The Dream implementation in dllm also hardcodes `is_causal=False`:

```python
attn_output = torch.nn.functional.scaled_dot_product_attention(
    query_states, key_states, value_states,
    attn_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
    dropout_p=self.attention_dropout if self.training else 0.0,
    is_causal=False,  # hard coded
)
```

And converts 2D padding masks to 4D boolean masks in the model forward:
```python
elif attention_mask.dim() == 2:
    attention_mask = torch.logical_and(
        attention_mask.unsqueeze(1).unsqueeze(-2),
        attention_mask.unsqueeze(2).unsqueeze(-1),
    ).to(torch.bool)
```

### Summary of Reference Approaches

| Implementation | FA2 Strategy | 4D Mask? | is_causal? |
|---------------|-------------|----------|------------|
| LLaDA (official) | Call `flash_attn_func` directly | No (None for FA) | `causal=False` explicit |
| dllm/Dream | PyTorch SDPA | Yes (4D bool) | `is_causal=False` hardcoded |
| dllm/LLaDA | Same as official LLaDA | No (None for FA) | `causal=False` explicit |
| Our backbone | HF pipeline with 4D zeros | Yes (4D float) | Relies on HF pipeline |

## Known Pitfalls and Risks

### 1. Silent Causal Fallback (CRITICAL)
Before PR #39707 (Aug 2025), `is_causal` was **always True** in `_flash_attention_forward` regardless of what was passed. The function did `kwargs.pop("is_causal", None)` and used `module.is_causal` (always True). If using transformers < the version containing this fix, FA2 would silently apply causal masking even with a 4D bidirectional mask.

### 2. Gemma3 Bidirectional Mask Bug (Issue #39389)
The `create_causal_mask` refactor broke Gemma3's bidirectional mask for image tokens -- the mask was not reaching the attention forward during inference. Fixed in PR #39396. Demonstrates that HF mask plumbing is fragile and breaks across versions.

### 3. torch.compile Incompatibility (Issue #42950)
`masking_utils.create_causal_mask` breaks `torch.compile(fullgraph=True)`. Our 4D mask approach bypasses this function's logic via early return, but the function is still called. This may or may not affect compilation.

### 4. 4D Mask + FA2 = Undefined Behavior
No reference implementation passes a 4D mask through HF's FA2 pipeline. The `_flash_attention_forward` function assumes 2D masks. Passing 4D:
- May crash on `_upad_input` (shape mismatch in `attention_mask.sum(dim=-1)`)
- May silently ignore the mask (if `attention_mask is not None` check passes but the unpad code handles it unexpectedly)
- Behavior varies by transformers version

## Recommended Approach

### Option A: Register Custom Attention Function -- Recommended

Use the `AttentionInterface.register()` API (transformers >= 5.x) to register a custom FA2 wrapper that forces `is_causal=False` and passes `attention_mask=None` to the FA2 kernel:

```python
from transformers import AttentionInterface
from transformers.integrations.flash_attention import flash_attention_forward as _orig_fa2

def bidirectional_flash_attention_forward(module, query, key, value, attention_mask=None, **kwargs):
    """FA2 wrapper that forces bidirectional attention (is_causal=False).

    Drops the 4D attention mask (FA2 can't use it) and relies solely on
    is_causal=False for bidirectional behavior. Padding is handled by
    passing the original 2D mask if available.
    """
    # If a 4D mask was passed, we can't use it with FA2 -- drop it
    if attention_mask is not None and attention_mask.dim() == 4:
        attention_mask = None
    kwargs["is_causal"] = False
    return _orig_fa2(module, query, key, value, attention_mask, **kwargs)

AttentionInterface.register("bidirectional_fa2", bidirectional_flash_attention_forward)

# Then load model with:
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", attn_implementation="bidirectional_fa2"
)
```

**Pros:** Clean, version-stable, no monkey-patching, follows HF's intended extension mechanism.
**Cons:** Requires transformers >= 5.x for `AttentionInterface`. Loses padding-aware unpadding optimization (minor for packed sequences with no padding).

### Option B: Direct flash_attn Call in Backbone -- Consider (What LLaDA Does)

Bypass HF's attention pipeline entirely. Override the attention computation at a lower level:

```python
from flash_attn import flash_attn_func

# Replace the attention forward in each layer
for layer in model.transformer.model.layers:
    original_attn = layer.self_attn
    # Wrap to call flash_attn_func with causal=False
```

**Pros:** Maximum control, proven by LLaDA.
**Cons:** Fragile to HF model changes, requires per-architecture implementation, loses HF's attention interface benefits.

### Option C: Keep 4D Mask, Use SDPA Backend -- Fallback

For `attn_implementation="sdpa"`, our current 4D mask approach works correctly. PyTorch's SDPA can dispatch to the FlashAttention kernel internally when `is_causal=False` and a compatible mask is provided.

```python
# In BidirectionalTransformer, when using SDPA:
# 4D all-zeros mask -> SDPA with is_causal=False -> may use FA kernel internally
```

**Pros:** Current code works as-is. No changes needed.
**Cons:** SDPA's internal kernel selection may not always pick FA2. Less control over which kernel runs. Performance may be slightly lower than direct FA2.

### Option D: Conditional Mask Strategy -- Pragmatic

Modify `BidirectionalTransformer.forward()` to detect the attention backend and adjust mask strategy:

```python
def forward(self, input_ids, attention_mask=None, **kwargs):
    B, L = input_ids.shape
    attn_impl = self.transformer.config._attn_implementation

    if attn_impl in ("flash_attention_2", "flash_attention_3"):
        # FA2: don't pass 4D mask, instead pass 2D padding mask (or None)
        # is_causal=False is handled via kwargs or custom attention fn
        if attention_mask is not None and attention_mask.dim() == 2:
            pass  # Keep 2D padding mask for FA2's unpadding optimization
        else:
            attention_mask = None  # FA2 + is_causal=False = bidirectional
        kwargs["is_causal"] = False  # Requires transformers with PR #39707
    else:
        # eager/sdpa: use 4D all-zeros mask as before
        attention_mask = self._build_4d_mask(input_ids, attention_mask)

    return self.transformer(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
```

**Pros:** Works across all backends, minimal code change.
**Cons:** `is_causal` kwarg propagation depends on the model architecture supporting `**kwargs` in attention layers. Not all models forward kwargs to attention.

## Recommended Stack

1. **Default (MPS/CPU):** Keep current `attn_implementation="eager"` with 4D all-zeros mask. Works correctly. No changes needed.

2. **GPU training (CUDA):** Use Option A (`AttentionInterface.register`) with a `bidirectional_fa2` attention function. Set `attn_implementation="bidirectional_fa2"` in `ModelConfig`. Requires transformers >= 5.x.

3. **Fallback GPU:** Use `attn_implementation="sdpa"` with our existing 4D mask. PyTorch's SDPA will handle it correctly and may internally dispatch to efficient kernels. Performance is close to FA2.

## Open Questions

1. **What transformers version includes PR #39707?** Merged Aug 12, 2025. Need to verify exact release tag. Our pin is `<5.10` which should include it.

2. **Does `AttentionInterface.register` work with `from_config` (random init)?** Need to test -- the register API may require model code to use `ALL_ATTENTION_FUNCTIONS` dispatch (which Qwen2/3 and LLaMA do in recent versions).

3. **Does `is_causal=False` kwarg propagate through all Qwen3 layers?** The attention interface passes kwargs, but we need to verify Qwen3's decoder layer forwards them correctly.

4. **Padding handling with FA2 + bidirectional:** When we drop the 4D mask for FA2, padding tokens get full bidirectional attention (they attend to and are attended by all tokens). For pretraining with packed sequences (no padding), this is fine. For inference with variable-length inputs, we need the 2D padding mask to flow through FA2's unpadding pipeline. Test this explicitly.

5. **Performance comparison:** SDPA with 4D mask vs. registered bidirectional FA2 vs. direct flash_attn_func call. Benchmark on Qwen3-0.6B with seq_len 512/1024/2048.

## Sources

- [HF Transformers Attention Interface Docs](https://huggingface.co/docs/transformers/attention_interface)
- [4D Masks Support Blog Post](https://huggingface.co/blog/poedator/4d-masks)
- [Issue #39554: is_causal not used in flash_attention_forward](https://github.com/huggingface/transformers/issues/39554)
- [PR #39707: Fix is_causal handling in flash_attention_forward](https://github.com/huggingface/transformers/pull/39707) (merged Aug 12, 2025)
- [Issue #39389: Gemma3 bidirectional mask bug](https://github.com/huggingface/transformers/issues/39389)
- [Issue #42950: create_causal_mask breaks torch.compile](https://github.com/huggingface/transformers/issues/42950)
- [masking_utils.py source](https://github.com/huggingface/transformers/blob/main/src/transformers/masking_utils.py)
- [modeling_flash_attention_utils.py source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_flash_attention_utils.py)
- [LLaDA-8B-Base Model (GSAI-ML)](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)
- [LLaDA Official Repo](https://github.com/ML-GSAI/LLaDA)
- [dllm Library (ZHZisZZ)](https://github.com/ZHZisZZ/dllm)
- [Qwen2 modeling source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py)
- [Qwen3 modeling source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)
- [Issue #29525: Custom 4D masks broken by attention refactor](https://github.com/huggingface/transformers/issues/29525)
