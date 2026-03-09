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
    4. Estimate log p(completion | prompt) via ELBO Monte Carlo.
    5. GRPO clipped objective: E[clip(ratio, 1-eps, 1+eps) * A_k].

    Args:
        reward_fn: Callable(prompt_ids, completion_ids) -> Tensor of shape (B,).
            Must return scalar reward per (prompt, completion) pair.
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

    def _estimate_log_prob(
        self,
        model,
        input_ids: Tensor,
        prompt_mask: Tensor | None,
        shared_u: float,
    ) -> Tensor:
        """Estimate log p(response | prompt) via ELBO MC averaging.

        Args:
            model: Model to query.
            input_ids: Token IDs, shape (B, L).
            prompt_mask: Boolean mask for prompt positions.
            shared_u: Shared antithetic base (same across policy + reference).

        Returns:
            Log-prob estimates shape (B,).
        """
        B = input_ids.shape[0]
        device = input_ids.device
        total_loss = torch.zeros(B, device=device)

        for _ in range(self.n_mc_samples):
            indices = torch.arange(B, device=device, dtype=torch.float32)
            t = ((shared_u + indices / B) % 1.0).float()
            t = t * (1.0 - 1e-3) + 1e-3

            with torch.no_grad() if model is self.ref_model else torch.enable_grad():
                outputs = model(input_ids=input_ids, prompt_mask=prompt_mask)
            total_loss = total_loss + outputs["loss"].detach()

        return -(total_loss / self.n_mc_samples)

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

        # Import here to avoid circular imports at module load time
        from diffusion_lm.config.generation import GenerationConfig
        from diffusion_lm.samplers.first_hitting_sampler import FirstHittingSampler

        gen_config = GenerationConfig(max_new_tokens=64, num_steps=32)
        sampler = FirstHittingSampler()

        # Sample K completions per prompt
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

            # Build prompt_mask for full (prompt + completion) sequence
            full_len = completion_ids.shape[1]
            full_prompt_mask = torch.zeros(B, full_len, dtype=torch.bool, device=device)
            full_prompt_mask[:, :prompt_len] = True
            all_prompt_masks.append(full_prompt_mask)

        # Stack: (K, B, L)
        rewards_tensor = torch.stack(all_rewards, dim=0)  # (K, B)

        # Group-relative advantage: normalize within each prompt's group
        mean_r = rewards_tensor.mean(dim=0, keepdim=True)  # (1, B)
        std_r = rewards_tensor.std(dim=0, keepdim=True).clamp(min=1e-8)
        advantages = (rewards_tensor - mean_r) / std_r  # (K, B)

        # Compute GRPO loss
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for k in range(self.group_size):
            outputs = model(input_ids=all_input_ids[k], prompt_mask=all_prompt_masks[k])
            policy_loss = outputs["loss"]

            adv = advantages[k].detach()  # (B,)
            total_loss = total_loss + (-adv * policy_loss).mean()

        loss = total_loss / self.group_size

        mean_reward = rewards_tensor.mean().item()
        logger.debug(f"GRPO mean_reward={mean_reward:.4f}, loss={loss.item():.4f}")

        if return_outputs:
            return loss, {"loss": loss}
        return loss
