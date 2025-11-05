import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from typing import cast, TypeVar
from tianshou.algorithm.algorithm_base import OnPolicyAlgorithm, TrainingStats
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import ReplayBuffer, SequenceSummaryStats
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import RolloutBatchProtocol, GrpoBatchProtocol

BatchT = TypeVar("BatchT", bound=BatchProtocol)


@dataclass(kw_only=True)
class GRPOTrainingStats(TrainingStats):
    """Training statistics for GRPO algorithm."""

    loss: SequenceSummaryStats
    clip_loss: SequenceSummaryStats
    kl_loss: SequenceSummaryStats
    gradient_steps: int


def _get_episode_start_indices(batch: BatchT) -> np.ndarray:
    """Get indices where new episodes start using done flags."""
    # First timestep is always a episode start
    is_start = np.zeros(len(batch), dtype=bool)
    is_start[0] = True
    # Timestep after a done is a episode start
    if len(batch) > 1:
        is_start[1:] = batch.done[:-1]
    return np.where(is_start)[0]


def get_unique_initial_states(batch: BatchT) -> dict[str, np.ndarray]:
    """Get mapping from state hash to actual state array."""
    start_indices = _get_episode_start_indices(batch)
    first_timesteps = batch[start_indices]

    unique_states = {}
    for state in first_timesteps.obs:  # Use obs directly, not initial_state
        state_hash = numpy_hash_rounded(state)
        if state_hash not in unique_states:
            unique_states[state_hash] = state

    return unique_states


def filter_by_initial_state(
    batch: GrpoBatchProtocol,
    state_hash: str,
) -> GrpoBatchProtocol:
    """Filter batch to only include timesteps from episodes with given initial state."""
    start_indices = _get_episode_start_indices(batch)
    first_timesteps = batch[start_indices]

    # Find matching episode IDs
    matching_episode_ids = []
    for i, start_idx in enumerate(start_indices):
        state = first_timesteps.obs[i]
        if numpy_hash_rounded(state) == state_hash:
            matching_episode_ids.append(batch.episode_id[start_idx])

    mask = np.isin(batch.episode_id, matching_episode_ids)
    return batch[mask]


def numpy_hash_rounded(arr: np.ndarray, decimals: int = 6, algorithm: str = "sha1") -> str:
    """Hash numpy array with rounding to handle floating-point precision."""
    h = hashlib.new(algorithm)
    # Round to specified decimals before hashing
    rounded = np.round(arr, decimals=decimals)
    h.update(rounded.tobytes(order="C"))
    h.update(arr.dtype.str.encode())
    h.update(str(arr.shape).encode())
    return h.hexdigest()


class GRPO(OnPolicyAlgorithm[ProbabilisticActorPolicy]):
    def __init__(
        self,
        policy: ProbabilisticActorPolicy,
        optim: OptimizerFactory,
        max_batchsize: int = 256,
        clip_epsilon: float = 0.2,
        kl_coefficient: float = 0.1,
        n_epochs: int = 20,
    ):
        super().__init__(policy=policy)
        self.optimizer = self._create_optimizer(self.policy, optim)
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.max_batchsize = max_batchsize
        self.kl_coefficient = kl_coefficient

    def _extract_episode_boundaries(
        self, batch: RolloutBatchProtocol
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract start and end indices for each episode in the batch.

        Returns:
            start_indices: Array of shape (num_episodes,) starting indices for each episode
            end_indices: Array of shape (num_episodes,) ending indices for each episode
        """
        end_indices = np.where(batch.done)[0]
        start_indices = np.concatenate(([0], end_indices[:-1] + 1))
        return start_indices, end_indices

    def _compute_episode_returns(
        self, batch: RolloutBatchProtocol, start_indices: np.ndarray, end_indices: np.ndarray
    ) -> np.ndarray:
        """Compute total return for each episode.

        Returns:
            episodic_returns: Array of shape (num_episodes,) with total returns
        """
        num_episodes = len(start_indices)
        episodic_returns = np.zeros(num_episodes, dtype=np.float64)

        for i in range(num_episodes):
            start_idx = start_indices[i]
            end_idx = end_indices[i]
            # Sum rewards from start to end (inclusive)
            episodic_returns[i] = batch.rew[start_idx : end_idx + 1].sum()

        return episodic_returns

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> GrpoBatchProtocol:
        """Preprocess batch by computing advantages using GRPO algorithm."""
        # Step 1: Extract episode boundaries
        start_indices, end_indices = self._extract_episode_boundaries(batch)
        num_episodes = len(start_indices)
        last_done_idx = end_indices[-1]

        # Dump last episode if incomplete
        if last_done_idx < len(batch) - 1:
            batch = batch[: last_done_idx + 1]

        # Step 2: Pre-allocate metadata arrays (compute once)
        batch.episode_id = np.zeros(len(batch), dtype=np.int64)
        batch.initial_state = np.empty((len(batch),) + batch.obs.shape[1:], dtype=batch.obs.dtype)
        batch.adv = np.zeros(len(batch), dtype=np.float64)

        # Step 3: Extract starting states and compute hashes ONCE
        starting_states = np.round(batch[start_indices].obs, decimals=6)
        state_hashes = np.array([numpy_hash_rounded(state) for state in starting_states])
        unique_hashes, inverse_indices = np.unique(state_hashes, return_inverse=True)

        # Step 4: Compute episode returns
        episodic_returns = self._compute_episode_returns(batch, start_indices, end_indices)

        # Step 5: Fill metadata arrays by episode
        for episode_id in range(num_episodes):
            start_idx = start_indices[episode_id]
            end_idx = end_indices[episode_id]
            initial_state = starting_states[episode_id]

            # Fill in-place by index range (like A2C does)
            batch.episode_id[start_idx : end_idx + 1] = episode_id
            batch.initial_state[start_idx : end_idx + 1] = initial_state

        # Step 6: Compute advantages per starting state group
        for i, unique_hash in enumerate(unique_hashes):
            mask = inverse_indices == i
            group_returns = episodic_returns[mask]

            mean_return = group_returns.mean()
            std_return = max(group_returns.std(), 1e-8)
            standardized_advantages = (group_returns - mean_return) / std_return

            # Assign advantages to all timesteps in each episode of this group
            for episode_id, advantage in zip(np.where(mask)[0], standardized_advantages):
                start_idx = start_indices[episode_id]
                end_idx = end_indices[episode_id]
                batch.adv[start_idx : end_idx + 1] = advantage

        # Step 7: Compute logp_old (convert to torch once)
        if isinstance(batch.act, np.ndarray):
            batch.act = torch.from_numpy(batch.act)
        if isinstance(batch.adv, np.ndarray):
            batch.adv = torch.from_numpy(batch.adv)

        logp_old = []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                logp_old.append(self.policy(minibatch).dist.log_prob(minibatch.act))
            batch.logp_old = torch.cat(logp_old, dim=0).flatten()

        return cast(GrpoBatchProtocol, cast(BatchProtocol, batch))

    def _update_with_batch(
        self,
        batch: GrpoBatchProtocol,
        batch_size: int | None,
        repeat: int,
    ) -> TrainingStats:
        """Update policy by iterating over initial states."""
        losses, clip_losses, kl_losses = [], [], []
        gradient_steps = 0
        split_batch_size = batch_size or self.max_batchsize

        # Get unique initial states
        unique_states = get_unique_initial_states(batch)

        # Iterate over each unique initial state
        for state_hash in unique_states.keys():
            # Filter to get all timesteps from episodes with this initial state
            state_batch = filter_by_initial_state(batch, state_hash)

            # Split and process minibatches
            for minibatch in state_batch.split(split_batch_size, shuffle=False, merge_last=True):
                gradient_steps += 1

                advantages = minibatch.adv
                dist = self.policy(minibatch).dist
                ratios = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                clipped_ratio = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

                episode_policy_loss = -torch.min(
                    ratios * advantages, clipped_ratio * advantages
                ).mean()

                inv_ratio = 1.0 / ratios
                kl_penalty = torch.mean(inv_ratio - torch.log(inv_ratio + 1e-8) - 1)

                loss = episode_policy_loss + self.kl_coefficient * kl_penalty
                self.optimizer.step(loss)

                losses.append(loss.item())
                clip_losses.append(episode_policy_loss.item())
                kl_losses.append(kl_penalty.item())

        loss_summary_stat = SequenceSummaryStats.from_sequence(losses)
        clip_losses_summary_stat = SequenceSummaryStats.from_sequence(clip_losses)
        kl_losses_summary_stat = SequenceSummaryStats.from_sequence(kl_losses)

        return GRPOTrainingStats(
            loss=loss_summary_stat,
            kl_loss=kl_losses_summary_stat,
            clip_loss=clip_losses_summary_stat,
            gradient_steps=gradient_steps,
        )
