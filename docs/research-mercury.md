# Research: Inception Labs Mercury (1, 2) — Diffusion LLM Architecture and Implications

## Executive Summary

Mercury 2 (Feb 24, 2026) is the first production-grade reasoning dLLM, achieving 1,009 tok/s on Blackwell GPUs — 5-10x faster than AR speed models (Haiku, GPT-5 Mini) at $0.25/$0.75 per M tokens. It uses **masked diffusion** on a standard Transformer backbone, trained **from scratch on trillions of tokens** (not AR-to-dLLM adaptation). Architecture internals are proprietary: no published parameter counts, noise schedules, step counts, or sampler details. The Mercury 1 paper (arXiv:2506.17298) is thin on methodology — the competitive advantage is engineering (custom CUDA kernels, adaptive stepping), not novel math. For our framework, Mercury validates the masked diffusion approach but offers no reproducible techniques beyond what LLaDA, Dream 7B, and the dLLM library already provide.

## Problem Statement

We need to understand what Mercury does differently so we can incorporate any transferable techniques into our diffusion LM training framework. Specifically: (1) should we change our diffusion process, schedule, or sampler based on Mercury's results? (2) are there architectural innovations we're missing? (3) how does Mercury's performance contextualize our framework's design decisions?

## Technology Evaluation

### Mercury 1 (Coder) — Jul 2025 — Consider (for reference only)

- **Paper:** arXiv:2506.17298 — "Mercury: Ultra-Fast Language Models Based on Diffusion"
- **Team:** Stanford/UCLA/Cornell researchers. CEO Stefano Ermon (co-inventor of diffusion methods). $50M funding from Menlo Ventures (Andrew Ng, Andrej Karpathy as angels).
- **Models:** Mercury Coder Mini (1,109 tok/s on H100), Mercury Coder Small (737 tok/s on H100)
- **Focus:** Code generation. Tied 2nd on Copilot Arena, surpassing GPT-4o Mini.
- **Architecture:** Transformer backbone, discrete masked diffusion, parallel token generation.
- **Training:** From scratch on trillions of tokens. NOT AR-to-dLLM adaptation.
- **Paper quality:** Deliberately vague on methodology. States "our methods extend [MDLM] through careful modifications" without specifying what those modifications are. No noise schedule, no step counts, no sampler pseudocode, no loss weighting details published.
- **Loss function:** Denoising objective `L(x) = -E_t[gamma(t) * E_{z_t~q} log p_theta(x|z_t)]` where `gamma(t) >= 0` is "user-specified." Specific weighting scheme not disclosed (likely ELBO-equivalent but unconfirmed).
- **Open source:** No. Closed weights, closed methodology. API-only.

### Mercury 2 — Feb 24, 2026 — Consider (for benchmarking context)

- **Release:** Feb 24, 2026. First reasoning-capable dLLM.
- **Speed:** 1,009 tok/s on Blackwell, ~674 tok/s measured by Artificial Analysis.
- **TTFT:** 3.21s (high — median is 1.83s). This is the diffusion "canvas initialization" cost.
- **Context window:** 128K tokens. Max output: 8,192 tokens (some sources say 50K — discrepancy).
- **Reasoning:** Tunable reasoning depth (think budget). Extended thinking mode.

**Benchmark results (Mercury 2):**

| Benchmark | Score | Context |
|-----------|-------|---------|
| AIME 2025 | 91.1% | Competitive with Haiku 4.5, GPT-5.2 Mini |
| GPQA Diamond | 73.6-74% | Rank ~73rd overall |
| LiveCodeBench | 67.3% | Rank ~18th |
| IFBench | 71.3% | Instruction following |
| SciCode | 38.4% | Scientific code |
| Tau2 Airline | 52.9% | Agent benchmark |
| AA Intelligence Index | 33/100 | Rank #23 of 134 models |

**Intelligence tier:** Haiku/GPT-Mini class. NOT Opus/Sonnet class. Positioned as "fast + cheap," not "smartest."

**Pricing:**

| | Input | Output |
|--|-------|--------|
| Mercury 2 | $0.25/M | $0.75/M |
| Gemini 3 Flash | $0.50/M | $3.00/M |
| Claude 4.5 Haiku | ~$0.80/M | ~$4.00/M |

**API:** OpenAI-compatible. `base_url = https://api.inceptionlabs.ai/v1`, model = `mercury-2`. Supports tool calling, JSON mode. Available on AWS Bedrock, Azure AI Foundry, OpenRouter.

**Known issues (from HN discussion and reviews):**
- Semantic reasoning failures (misunderstanding context in multi-step reasoning)
- Hallucination patterns: spelling errors like "Dieadona" instead of "Maradona"
- Inference glitches / looping (team acknowledges, working on fixes)
- High verbosity: generates ~3.5x median output tokens (69M vs 20M median in eval)
- TTFT is high relative to AR models — the "canvas" must be initialized

### Mercury 3 — Not announced

No Mercury 3 or Mercury Pro/Large found as of March 10, 2026. Mercury 2 is the latest.

## Architecture Patterns Found

### What Mercury Confirms About Our Approach

1. **Masked diffusion is the winning process type.** Mercury, LLaDA, Dream 7B, and the dLLM unified library all use masked (absorbing-state) diffusion. Not continuous, not SEDD. Our `MaskedDiffusionProcess` is the right default.

2. **Standard Transformer backbone works.** Mercury uses vanilla Transformers — no custom architecture. This validates our `BidirectionalTransformer` wrapper approach.

3. **Parallel generation is the speed advantage.** The core speed gain comes from generating N tokens per forward pass (where N = sequence length) in ~8-20 denoising steps, vs. N sequential forward passes for AR. Our framework already supports this via all samplers.

4. **Confidence-ordered unmasking during inference.** Mercury unmasks high-confidence tokens first, giving later steps better context. This aligns with standard categorical sampling in masked diffusion — NOT the First-Hitting Sampler. However, the FHS achieves the same effect with theoretical correctness guarantees.

5. **Custom CUDA kernels matter for production speed.** Mercury's 1000+ tok/s comes from custom inference kernels, not algorithmic innovation. This is engineering, not research.

### What Mercury Does Differently (That We Cannot Reproduce)

1. **Trained from scratch on trillions of tokens.** Mercury was NOT adapted from an AR checkpoint. The paper says "trillions of tokens" of training data including "web crawls along with carefully curated real and synthetic datasets." This is a massive compute investment we cannot replicate. Our AR-to-dLLM adaptation path (validated by Dream 7B at ~500x less compute) remains the correct choice for our resource level.

2. **Proprietary inference engine.** Custom CUDA kernels for diffusion sampling. Not reproducible without their codebase.

3. **Adaptive denoising steps.** One source claims "simple structured outputs may need 8 steps while complex reasoning uses 16-20." This is interesting — dynamically choosing step count based on output complexity. Our framework uses a fixed step count. **This is worth implementing.**

4. **Proprietary noise schedule and loss weighting.** The `gamma(t)` weighting function is not disclosed. Could be ELBO weighting (what we use), uniform, or something custom.

### Dream 7B — The Open Alternative (Aug 2025)

Dream 7B (arXiv:2508.15487) is the closest open-source analog to Mercury:
- **Architecture:** Masked diffusion on Qwen2.5-7B backbone (AR-to-dLLM adaptation)
- **Training:** 580B tokens of mixed corpus
- **Key innovation:** Context-adaptive token-level noise rescheduling — dynamically reassigns noise level per token based on corrupted context
- **Open source:** Yes (weights + code)
- **Performance:** Outperforms all open dLLMs on general, math, and coding tasks
- **Relevance:** Dream's context-adaptive rescheduling is the most interesting transferable technique from any 2026 dLLM work

## Key APIs and Services

### Inception API (Mercury 2)

| Property | Value |
|----------|-------|
| Endpoint | `https://api.inceptionlabs.ai/v1` |
| Protocol | OpenAI-compatible (chat completions) |
| Auth | API key (Bearer token) |
| Models | `mercury-2` |
| Context | 128K tokens |
| Max output | 8,192 tokens |
| Input price | $0.25/M tokens |
| Output price | $0.75/M tokens |
| Tool calling | Yes |
| JSON mode | Yes |
| Streaming | Yes |
| Rate limits | Not published |
| Regions | US (direct), also via AWS Bedrock, Azure AI Foundry |

**Also available via:** OpenRouter (`inception/mercury-2`), AWS Bedrock Marketplace, Azure AI Foundry.

## Known Pitfalls and Risks

1. **Mercury is closed-source.** No weights, no training code, no sampler code. Cannot be used for research reproduction. API-only.

2. **High TTFT.** 3.21s median TTFT vs 1.83s AR median. The diffusion "canvas initialization" adds latency before first token. Bad for streaming UX in chat applications.

3. **Verbosity problem.** Mercury 2 generates 3.5x more tokens than median models for the same tasks. This inflates output costs despite low per-token pricing.

4. **Max output is only 8K tokens.** Despite 128K context, output is capped at 8,192 tokens. This limits long-form generation use cases.

5. **No multimodal support.** Text-only. No vision, audio, or video input/output.

6. **"From scratch" training is not transferable.** Mercury's approach requires trillions of tokens of compute. For teams without that budget, AR-to-dLLM adaptation (Dream 7B, LLaDA, Fast-dLLM v2) is the practical path.

7. **Paper is deliberately opaque.** The Mercury paper (arXiv:2506.17298) withholds key details. You cannot reproduce their results from the paper alone.

## Recommended Stack (Updated)

No changes to our framework's core architecture based on Mercury findings. Mercury validates our existing choices:

- **Diffusion process:** Masked (absorbing-state) diffusion -- confirmed correct
- **Backbone:** Standard Transformer with bidirectional attention via explicit 4D mask -- confirmed correct
- **Init path:** AR-to-dLLM adaptation -- still optimal for our compute budget (Mercury's from-scratch path is infeasible)
- **Loss:** ELBO-weighted CE -- consistent with Mercury's `gamma(t)` formulation
- **Sampler:** First-Hitting Sampler -- theoretically superior to Mercury's confidence-based unmasking

**Two features worth adding based on Mercury + Dream 7B research:**

1. **Adaptive denoising step count** — Mercury uses 8 steps for simple outputs, 16-20 for complex reasoning. Implement as `GenerationConfig.adaptive_steps=True` with a confidence threshold to determine early stopping.

2. **Context-adaptive noise rescheduling** (from Dream 7B, not Mercury) — Dynamically reassign per-token noise level based on corrupted context. This is the most interesting open technique from 2026 dLLM work. See arXiv:2508.15487 Section 3.2.

## Open Questions

1. **What is Mercury's actual noise schedule?** The paper says `gamma(t) >= 0` but doesn't specify. If it's not ELBO weighting, what is it? No way to find out without reverse engineering.

2. **How many parameters is Mercury 2?** Undisclosed. Likely in the 7-14B range based on speed characteristics and intelligence tier positioning (Haiku-class).

3. **Does Mercury use a novel sampler?** They describe "confidence-ordered unmasking" which sounds like standard categorical sampling. The FHS should be strictly better. But Mercury may have proprietary improvements.

4. **Will Inception open-source anything?** No indication. Their competitive advantage is engineering (custom kernels), not algorithmic novelty. Open-sourcing weights would invite competitors to serve them faster.

5. **Max output discrepancy:** llm-stats.com says 8,192 tokens max output, but Artificial Analysis mentions 50K. Needs clarification from Inception docs.

6. **Dream 7B's context-adaptive rescheduling:** How much does this help in practice? Is the improvement from rescheduling or just from using Qwen2.5 as init? Need ablation data.

## Sources

- [Mercury 1 Paper (arXiv:2506.17298)](https://arxiv.org/abs/2506.17298)
- [Mercury 1 Paper — HTML version](https://arxiv.org/html/2506.17298v1)
- [Introducing Mercury 2 — Inception Blog](https://www.inceptionlabs.ai/blog/introducing-mercury-2)
- [Mercury 2 Launch Press Release — BusinessWire](https://www.businesswire.com/news/home/20260224034496/en/Inception-Launches-Mercury-2-the-Fastest-Reasoning-LLM-5x-Faster-Than-Leading-Speed-Optimized-LLMs-with-Dramatically-Lower-Inference-Cost)
- [Mercury 2 — Artificial Analysis Evaluation](https://artificialanalysis.ai/models/mercury-2)
- [Mercury 2 — LLM Stats](https://llm-stats.com/models/mercury-2)
- [Mercury 2 — OpenRouter](https://openrouter.ai/inception/mercury-2)
- [Mercury 2 HN Discussion](https://news.ycombinator.com/item?id=47144464)
- [Mercury 2 Speed Guide — DigitalApplied](https://www.digitalapplied.com/blog/inception-labs-mercury-2-diffusion-llm-speed-guide)
- [Mercury on The New Stack](https://thenewstack.io/inception-labs-mercury-2-diffusion/)
- [Scaling Mercury Blog Post](https://www.inceptionlabs.ai/blog/mercury-refreshed)
- [Mercury on AWS Bedrock](https://aws.amazon.com/blogs/machine-learning/mercury-foundation-models-from-inception-labs-are-now-available-in-amazon-bedrock-marketplace-and-amazon-sagemaker-jumpstart/)
- [Inception API — Tool Use Docs](https://docs.inceptionlabs.ai/capabilities/tool-use)
- [Dream 7B Paper (arXiv:2508.15487)](https://arxiv.org/abs/2508.15487)
- [Dream 7B GitHub](https://github.com/DreamLM/Dream)
- [dLLM Unified Library (arXiv:2602.22661)](https://arxiv.org/abs/2602.22661)
- [The Decoder — Mercury 2 Coverage](https://the-decoder.com/inception-launches-mercury-2-the-first-diffusion-based-language-reasoning-model/)
- [Emergent Mind — Mercury Analysis](https://www.emergentmind.com/papers/2506.17298)
