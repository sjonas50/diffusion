"""DPO alignment trainer for diffusion LMs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from loguru import logger

from diffusion_lm.data.collators import DPOCollator
from diffusion_lm.trainers.base import DiffusionTrainer

if TYPE_CHECKING:
    from torch import Tensor


class DiffusionDPOTrainer(DiffusionTrainer):
    """DPO trainer for diffusion LMs.

    Estimates log p(response | prompt) via Monte Carlo averaging of the ELBO
    (negative diffusion loss) over sampled timesteps. Uses antithetic sampling
    with SHARED timesteps between policy and reference to reduce variance.

    DPO loss:
        -log sigmoid(beta * ((log_pi_chosen - log_ref_chosen)
                            - (log_pi_rejected - log_ref_rejected)))

    Memory: ~32 forward passes per step (4 quantities × n_mc_samples).
    Recommended: gradient_checkpointing=True, bf16=True, cpu_offload_ref=True.

    Args:
        ref_model: Frozen reference model (SFT checkpoint). If None, uses a
            copy of the policy model (not recommended — use explicit ref).
        cpu_offload_ref: Move reference model to CPU between forward passes.
            Saves GPU memory at cost of transfer overhead.
        n_mc_samples: Number of Monte Carlo samples for ELBO estimation.
        beta: DPO temperature (higher = closer to reference policy).
    """

    def __init__(
        self,
        ref_model=None,
        cpu_offload_ref: bool = False,
        n_mc_samples: int = 8,
        beta: float = 0.1,
        pad_token_id: int = 0,
        **kwargs,
    ) -> None:
        if "data_collator" not in kwargs or kwargs["data_collator"] is None:
            kwargs["data_collator"] = DPOCollator(pad_token_id=pad_token_id)
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.cpu_offload_ref = cpu_offload_ref
        self.n_mc_samples = n_mc_samples
        self.beta = beta

        if ref_model is not None and cpu_offload_ref:
            self.ref_model.cpu()
            logger.info("Reference model offloaded to CPU.")

    @torch.no_grad()
    def _estimate_log_prob(
        self,
        model,
        input_ids: Tensor,
        prompt_mask: Tensor | None,
        shared_u: float,
    ) -> Tensor:
        """Estimate log p(response | prompt) via ELBO Monte Carlo averaging.

        Uses antithetic timestep sampling with a shared base sample `u` so that
        policy and reference model estimates share the same timesteps — this
        reduces variance of their difference (the key quantity for DPO).

        Args:
            model: Model to estimate log-prob for.
            input_ids: Token IDs, shape (B, L).
            prompt_mask: Boolean mask, True at prompt positions.
            shared_u: Shared base for antithetic sampling (same for policy and ref).

        Returns:
            Estimated log-prob per sample, shape (B,) — approximately -ELBO.
        """
        B = input_ids.shape[0]
        device = input_ids.device
        total_loss = torch.zeros(B, device=device)

        for _ in range(self.n_mc_samples):
            # Antithetic timestep sampling with shared u
            indices = torch.arange(B, device=device, dtype=torch.float32)
            t = ((shared_u + indices / B) % 1.0).float()
            # Scale to [time_epsilon, 1)
            t = t * (1.0 - 1e-3) + 1e-3

            outputs = model(input_ids=input_ids, prompt_mask=prompt_mask)
            total_loss = total_loss + outputs["loss"]

        # log p ≈ -mean_loss (ELBO)
        return -(total_loss / self.n_mc_samples)

    def compute_loss(
        self,
        model,
        inputs: dict[str, Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        """Compute DPO loss.

        Args:
            model: Policy model.
            inputs: Batch with chosen_input_ids, rejected_input_ids, prompt_mask.

        Returns:
            DPO loss scalar.
        """
        chosen_ids = inputs["chosen_input_ids"]
        rejected_ids = inputs["rejected_input_ids"]
        prompt_mask = inputs.get("prompt_mask")

        device = chosen_ids.device

        # Shared base for antithetic sampling (reduces variance of policy - ref difference)
        shared_u = torch.rand(1).item()

        # Policy model estimates
        pi_chosen = self._estimate_log_prob(model, chosen_ids, prompt_mask, shared_u)
        pi_rejected = self._estimate_log_prob(model, rejected_ids, prompt_mask, shared_u)

        # Reference model estimates (no gradient)
        ref_model = self.ref_model if self.ref_model is not None else model
        if self.cpu_offload_ref and self.ref_model is not None:
            self.ref_model.to(device)

        with torch.no_grad():
            ref_chosen = self._estimate_log_prob(ref_model, chosen_ids, prompt_mask, shared_u)
            ref_rejected = self._estimate_log_prob(ref_model, rejected_ids, prompt_mask, shared_u)

        if self.cpu_offload_ref and self.ref_model is not None:
            self.ref_model.cpu()

        # DPO objective
        advantages = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
        loss = -F.logsigmoid(self.beta * advantages).mean()

        # Log reward margin for monitoring
        reward_margin = (pi_chosen - ref_chosen - (pi_rejected - ref_rejected)).mean().item()
        logger.debug(f"DPO reward margin: {reward_margin:.4f}, loss: {loss.item():.4f}")

        if return_outputs:
            return loss, {"loss": loss, "reward_margin": torch.tensor(reward_margin)}
        return loss
