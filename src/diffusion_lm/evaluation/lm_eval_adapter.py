"""lm-eval harness adapter for diffusion LMs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    pass


class DiffusionLMEvalAdapter:
    """Adapter implementing the lm-eval API for diffusion LMs.

    Wraps MaskedDiffusionLM to be compatible with the EleutherAI lm-evaluation-harness
    (https://github.com/EleutherAI/lm-evaluation-harness).

    Implements:
        - loglikelihood(requests): ELBO MC estimation with n_samples=32.
        - generate_until(requests): FirstHittingSampler + stop sequence truncation.

    Args:
        model: MaskedDiffusionLM instance.
        tokenizer: HuggingFace tokenizer.
        batch_size: Inference batch size.
        n_loglikelihood_samples: MC samples for log-likelihood estimation.
        device: Compute device string.
    """

    def __init__(
        self,
        model,
        tokenizer,
        batch_size: int = 8,
        n_loglikelihood_samples: int = 32,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.n_loglikelihood_samples = n_loglikelihood_samples
        self._device = device

        self.model.eval()
        self.model.to(device)

    def loglikelihood(self, requests: list) -> list[tuple[float, bool]]:
        """Estimate log-likelihood for (context, continuation) pairs.

        Args:
            requests: List of (context_str, continuation_str) tuples.

        Returns:
            List of (log_prob, is_greedy) tuples. is_greedy is always False
            for diffusion LMs (ELBO estimation is not exact).
        """
        import torch

        results = []
        for context, continuation in requests:
            full_text = context + continuation
            full_enc = self.tokenizer(full_text, return_tensors="pt")
            ctx_enc = self.tokenizer(context, return_tensors="pt")

            input_ids = full_enc["input_ids"].to(self._device)
            prompt_len = ctx_enc["input_ids"].shape[1]
            full_len = input_ids.shape[1]

            # Build prompt mask: True for context positions
            prompt_mask = torch.zeros(1, full_len, dtype=torch.bool, device=self._device)
            prompt_mask[:, :prompt_len] = True

            # MC estimate of ELBO log-likelihood over continuation
            total_loss = 0.0
            valid_samples = 0
            with torch.no_grad():
                for _ in range(self.n_loglikelihood_samples):
                    outputs = self.model(input_ids=input_ids, prompt_mask=prompt_mask)
                    loss = outputs["loss"]
                    if not loss.isnan():
                        total_loss += loss.item()
                        valid_samples += 1

            if valid_samples == 0:
                log_prob = -float("inf")
            else:
                avg_nll = total_loss / valid_samples
                log_prob = -avg_nll * (full_len - prompt_len)

            results.append((log_prob, False))
            logger.debug(f"loglikelihood: log_prob={log_prob:.4f}")

        return results

    def generate_until(self, requests: list) -> list[str]:
        """Generate text until stop sequences for each request.

        Args:
            requests: List of (context_str, gen_kwargs) tuples where gen_kwargs
                contains "until" (list of stop strings) and "max_gen_toks".

        Returns:
            List of generated continuation strings (without the context).
        """
        import torch

        from diffusion_lm.config.generation import GenerationConfig
        from diffusion_lm.samplers.first_hitting_sampler import FirstHittingSampler

        sampler = FirstHittingSampler()
        results = []

        for context, gen_kwargs in requests:
            stop_sequences = gen_kwargs.get("until", [])
            max_gen_toks = gen_kwargs.get("max_gen_toks", 128)

            enc = self.tokenizer(context, return_tensors="pt")
            prompt_ids = enc["input_ids"].to(self._device)

            gen_config = GenerationConfig(
                max_new_tokens=max_gen_toks,
                num_steps=64,
                running_confidence_remasking=True,
            )

            with torch.no_grad():
                output = sampler.generate(self.model, prompt_ids, gen_config)

            prompt_len = prompt_ids.shape[1]
            generated_ids = output.sequences[:, prompt_len:][0]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Truncate at stop sequences
            for stop in stop_sequences:
                if stop in generated_text:
                    generated_text = generated_text[: generated_text.index(stop)]

            results.append(generated_text)
            logger.debug(f"generate_until: generated {len(generated_text)} chars")

        return results
