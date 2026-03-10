"""ELBO-based perplexity evaluation for diffusion LMs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from loguru import logger

if TYPE_CHECKING:
    pass


class ELBOPerplexity:
    """Compute ELBO perplexity bound for masked diffusion LMs.

    Uses Monte Carlo estimation of the ELBO (same as training objective) to
    compute a perplexity upper bound. Using First-Hitting semantics (rather than
    categorical sampling) gives honest PPL comparable to AR model evaluation.

    Args:
        num_timestep_samples: MC samples per batch for timestep integration.
        batch_size: Evaluation batch size.
        pad_token_id: Token ID used for padding (to exclude from token count).
    """

    def __init__(
        self,
        num_timestep_samples: int = 32,
        batch_size: int = 8,
        pad_token_id: int = 0,
    ) -> None:
        self.num_timestep_samples = num_timestep_samples
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id

    def compute(
        self,
        model,
        dataset,
        device: str = "cpu",
    ) -> dict[str, float]:
        """Compute ELBO perplexity bound over the dataset.

        Args:
            model: MaskedDiffusionLM.
            dataset: Iterable of dicts with "input_ids" key.
            device: Compute device.

        Returns:
            Dict with "ppl_bound" and "nll_per_token".
        """
        from torch.utils.data import DataLoader

        model.eval()
        model.to(device)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        total_nll = 0.0
        total_tokens = 0

        logger.info("Computing ELBO perplexity...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                B, L = input_ids.shape

                batch_nll = 0.0
                for _ in range(self.num_timestep_samples):
                    outputs = model(input_ids=input_ids)
                    # loss is mean NLL per masked token (ELBO-weighted)
                    loss = outputs["loss"]
                    if not loss.isnan():
                        batch_nll += loss.item()

                avg_nll = batch_nll / self.num_timestep_samples
                # Count non-pad tokens (works regardless of pad_token_id value)
                n_tokens = (input_ids != self.pad_token_id).sum().item()
                total_nll += avg_nll * n_tokens
                total_tokens += n_tokens

                if (batch_idx + 1) % 10 == 0:
                    running_ppl = torch.exp(torch.tensor(total_nll / max(total_tokens, 1))).item()
                    logger.debug(f"Batch {batch_idx + 1}: running PPL bound = {running_ppl:.2f}")

        if total_tokens == 0:
            logger.warning("No tokens evaluated — returning PPL=inf")
            return {"ppl_bound": float("inf"), "nll_per_token": float("inf")}

        nll_per_token = total_nll / total_tokens
        ppl_bound = torch.exp(torch.tensor(nll_per_token)).item()

        logger.info(f"ELBO Perplexity bound: {ppl_bound:.2f} (NLL/tok: {nll_per_token:.4f})")
        return {"ppl_bound": ppl_bound, "nll_per_token": nll_per_token}
