import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from typing import cast
from tianshou.algorithm.algorithm_base import OnPolicyAlgorithm, TrainingStats
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import ReplayBuffer, SequenceSummaryStats
from tianshou.data.types import RolloutBatchProtocol, TrajectoryBatchProtocol


@dataclass(kw_only=True)
class GRPOTrainingStats(TrainingStats):
    """Training statistics for GRPO algorithm."""

    loss: SequenceSummaryStats
    clip_loss: SequenceSummaryStats
    kl_loss: SequenceSummaryStats
    gradient_steps: int


def get_unique_initial_states(batch: TrajectoryBatchProtocol) -> dict[str, np.ndarray]:
    """Get mapping from state hash to actual state array."""
    first_occurrence_mask = np.concatenate(
        [[True], batch.trajectory_id[1:] != batch.trajectory_id[:-1]]
    )
    first_timesteps = batch[first_occurrence_mask]

    unique_states = {}
    for state in first_timesteps.initial_state:
        state_hash = numpy_hash_rounded(state)
        if state_hash not in unique_states:
            unique_states[state_hash] = state

    return unique_states


def filter_by_initial_state(
    batch: TrajectoryBatchProtocol,
    state_hash: str,
) -> TrajectoryBatchProtocol:
    """Filter batch to only include timesteps from trajectories with given initial state."""
    # Find all trajectory IDs that match this initial state
    first_occurrence_mask = np.concatenate(
        [[True], batch.trajectory_id[1:] != batch.trajectory_id[:-1]]
    )
    first_timesteps_per_traj = batch[first_occurrence_mask]

    # Hash the initial states and find matching trajectory IDs
    matching_traj_ids = []
    for i, traj_id in enumerate(first_timesteps_per_traj.trajectory_id):
        state = first_timesteps_per_traj.initial_state[i]
        if numpy_hash_rounded(state) == state_hash:
            matching_traj_ids.append(traj_id)

    # Filter batch to only include these trajectory IDs
    mask = np.isin(batch.trajectory_id, matching_traj_ids)
    return cast(TrajectoryBatchProtocol, batch[mask])


def numpy_hash_rounded(arr: np.ndarray, decimals: int = 6, algorithm="sha1") -> str:
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
        clip_epsilon=0.2,
        kl_coefficient=0.1,
        n_epochs=20,
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

    def _compute_trajectory_returns(
        self, batch: RolloutBatchProtocol, start_indices: np.ndarray, end_indices: np.ndarray
    ) -> np.ndarray:
        """Compute total return for each trajectory.

        Returns:
            trajectory_returns: Array of shape (num_episodes,) with total returns
        """
        num_episodes = len(start_indices)
        trajectory_returns = np.zeros(num_episodes, dtype=np.float64)

        for i in range(num_episodes):
            start_idx = start_indices[i]
            end_idx = end_indices[i]
            # Sum rewards from start to end (inclusive)
            trajectory_returns[i] = batch.rew[start_idx : end_idx + 1].sum()

        return trajectory_returns

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> TrajectoryBatchProtocol:
        """Preprocess batch by computing advantages using GRPO algorithm."""

        # Step 1: Extract episode boundaries
        start_indices, end_indices = self._extract_episode_boundaries(batch)
        num_episodes = len(start_indices)
        last_done_idx = end_indices[-1]

        # Dump last trajectory if incomplete
        if last_done_idx < len(batch) - 1:
            batch = batch[: last_done_idx + 1]

        # Step 2: Pre-allocate metadata arrays (compute once)
        batch.trajectory_id = np.zeros(len(batch), dtype=np.int64)
        batch.initial_state = np.empty((len(batch),) + batch.obs.shape[1:], dtype=batch.obs.dtype)
        batch.adv = np.zeros(len(batch), dtype=np.float64)

        # Step 3: Extract starting states and compute hashes ONCE
        starting_states = np.round(batch[start_indices].obs, decimals=6)
        state_hashes = np.array([numpy_hash_rounded(state) for state in starting_states])
        unique_hashes, inverse_indices = np.unique(state_hashes, return_inverse=True)

        # Step 4: Compute trajectory returns
        trajectory_returns = self._compute_trajectory_returns(batch, start_indices, end_indices)

        # Step 5: Fill metadata arrays by trajectory
        for traj_idx in range(num_episodes):
            start_idx = start_indices[traj_idx]
            end_idx = end_indices[traj_idx]
            initial_state = starting_states[traj_idx]

            # Fill in-place by index range (like A2C does)
            batch.trajectory_id[start_idx : end_idx + 1] = traj_idx
            batch.initial_state[start_idx : end_idx + 1] = initial_state

        # Step 6: Compute advantages per starting state group
        for i, unique_hash in enumerate(unique_hashes):
            mask = inverse_indices == i
            group_returns = trajectory_returns[mask]

            mean_return = group_returns.mean()
            std_return = max(group_returns.std(), 1e-8)
            standardized_advantages = (group_returns - mean_return) / std_return

            # Assign advantages to all timesteps in each trajectory of this group
            for traj_idx, advantage in zip(np.where(mask)[0], standardized_advantages):
                start_idx = start_indices[traj_idx]
                end_idx = end_indices[traj_idx]
                batch.adv[start_idx : end_idx + 1] = advantage

        # Compute logp_old
        # Ensure actions are torch tensors
        if isinstance(batch.act, np.ndarray):
            batch.act = torch.from_numpy(batch.act)
        if isinstance(batch.adv, np.ndarray):
            batch.adv = torch.from_numpy(batch.adv)

        logp_old = []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                logp_old.append(self.policy(minibatch).dist.log_prob(minibatch.act))
            batch.logp_old = torch.cat(logp_old, dim=0).flatten()

        return batch

    def _update_with_batch(
        self,
        batch: TrajectoryBatchProtocol,
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
            # Filter to get all timesteps from trajectories with this initial state
            state_batch = filter_by_initial_state(batch, state_hash)

            # Split and process minibatches
            for minibatch in state_batch.split(split_batch_size, shuffle=False, merge_last=True):
                gradient_steps += 1

                advantages = minibatch.adv
                dist = self.policy(minibatch).dist
                ratios = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                clipped_ratio = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

                trajectory_policy_loss = -torch.min(
                    ratios * advantages, clipped_ratio * advantages
                ).mean()

                inv_ratio = 1.0 / ratios
                trajectory_kl_penalty = torch.mean(inv_ratio - torch.log(inv_ratio + 1e-8) - 1)

                loss = trajectory_policy_loss + self.kl_coefficient * trajectory_kl_penalty
                self.optimizer.step(loss)

                losses.append(loss.item())
                clip_losses.append(trajectory_policy_loss.item())
                kl_losses.append(trajectory_kl_penalty.item())

        loss_summary_stat = SequenceSummaryStats.from_sequence(losses)
        clip_losses_summary_stat = SequenceSummaryStats.from_sequence(clip_losses)
        kl_losses_summary_stat = SequenceSummaryStats.from_sequence(kl_losses)

        return GRPOTrainingStats(
            loss=loss_summary_stat,
            kl_loss=kl_losses_summary_stat,
            clip_loss=clip_losses_summary_stat,
            gradient_steps=gradient_steps,
        )
