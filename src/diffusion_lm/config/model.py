"""Model configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the backbone model.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        init_from_pretrained: If True (default), load pretrained AR weights for
            AR-to-dLLM adaptation (~500x less compute than from scratch).
        attn_implementation: Attention backend. Use "flash_attention_2" on GPU,
            "eager" or "sdpa" on CPU/MPS.
        dtype: Model dtype string ("bfloat16", "float16", "float32").
        use_lora: Enable LoRA parameter-efficient fine-tuning.
        lora_rank: LoRA rank (r).
        lora_alpha: LoRA scaling factor.
        lora_target_modules: Module names to apply LoRA to. None = auto-detect.
    """

    model_name_or_path: str = "gpt2"
    init_from_pretrained: bool = True
    attn_implementation: str = "eager"
    dtype: str = "bfloat16"
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_target_modules: list[str] | None = field(
        default=None,
        metadata={
            "help": "LoRA target modules. None = auto-detect. "
            "For Qwen3/LLaMA: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        },
    )
