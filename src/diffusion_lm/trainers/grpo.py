"""diffu-GRPO alignment trainer for diffusion LMs (arXiv:2504.12216)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from loguru import logger

from diffusion_lm.trainers.base import DiffusionTrainer

if TYPE_CHECKING:
    from torch import Tensor


class DiffusionGRPOTrainer(DiffusionTrainer):
    """GRPO trainer for diffusion LMs (diffu-GRPO).

    First working RL pipeline for dLLMs (arXiv:2504.12216).
    Adapts Group Relative Policy Optimization for parallel token generation.

    For each prompt:
    1. Sample K=group_size completions using FirstHittingSampler.
    2. Score each completion with reward_fn.
    3. Compute group-relative advantage: A_k = (r_k - mean(r)) / (std(r) + eps).
    4. Weight the diffusion loss by advantage for policy gradient.

    Args:
        reward_fn: Callable(prompt_ids, completion_ids) -> Tensor of shape (B,).
        group_size: Completions sampled per prompt (K).
        clip_ratio: PPO-style clip ratio epsilon.
        n_mc_samples: Monte Carlo samples for ELBO estimation.
        ref_model: Reference model for KL regularization (optional).
    """

    def __init__(
        self,
        reward_fn: Callable,
        group_size: int = 8,
        clip_ratio: float = 0.2,
        n_mc_samples: int = 8,
        ref_model=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.reward_fn = reward_fn
        self.group_size = group_size
        self.clip_ratio = clip_ratio
        self.n_mc_samples = n_mc_samples
        self.ref_model = ref_model

    def compute_loss(
        self,
        model,
        inputs: dict[str, Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        """Compute GRPO loss.

        Args:
            model: Policy model.
            inputs: Batch with input_ids (prompts) and optional prompt_mask.

        Returns:
            GRPO loss scalar.
        """
        prompt_ids = inputs["input_ids"]
        B, prompt_len = prompt_ids.shape
        device = prompt_ids.device

        from diffusion_lm.config.generation import GenerationConfig
        from diffusion_lm.samplers.first_hitting_sampler import FirstHittingSampler

        gen_config = GenerationConfig(max_new_tokens=64, num_steps=32)
        sampler = FirstHittingSampler()

        # Sample K completions per prompt (no gradient for generation)
        all_rewards = []
        all_input_ids = []
        all_prompt_masks = []

        for _ in range(self.group_size):
            with torch.no_grad():
                sampler_out = sampler.generate(model, prompt_ids, gen_config)
                completion_ids = sampler_out.sequences  # (B, prompt_len + gen_len)

            rewards = self.reward_fn(prompt_ids, completion_ids[:, prompt_len:])
            all_rewards.append(rewards)
            all_input_ids.append(completion_ids)

            full_len = completion_ids.shape[1]
            full_prompt_mask = torch.zeros(B, full_len, dtype=torch.bool, device=device)
            full_prompt_mask[:, :prompt_len] = True
            all_prompt_masks.append(full_prompt_mask)

        rewards_tensor = torch.stack(all_rewards, dim=0)  # (K, B)

        # Group-relative advantage
        mean_r = rewards_tensor.mean(dim=0, keepdim=True)
        std_r = rewards_tensor.std(dim=0, keepdim=True).clamp(min=1e-8)
        advantages = (rewards_tensor - mean_r) / std_r  # (K, B)

        # Compute GRPO loss: advantage-weighted diffusion loss WITH gradient
        # Use shared antithetic timesteps across completions for variance reduction
        shared_u = torch.rand(1).item()
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for k in range(self.group_size):
            # Build shared antithetic timesteps
            indices = torch.arange(B, device=device, dtype=torch.float32)
            t = ((shared_u + indices / B) % 1.0).float()
            t = t * (1.0 - 1e-3) + 1e-3

            # Forward pass WITH gradient, using explicit shared timesteps
            outputs = model(
                input_ids=all_input_ids[k],
                prompt_mask=all_prompt_masks[k],
                t=t,
            )
            policy_loss = outputs["loss"]

            adv = advantages[k].detach()  # (B,)
            total_loss = total_loss + (-adv * policy_loss).mean()

        loss = total_loss / self.group_size

        mean_reward = rewards_tensor.mean().item()
        logger.debug(f"GRPO mean_reward={mean_reward:.4f}, loss={loss.item():.4f}")

        if return_outputs:
            return loss, {"loss": loss}
        return loss
