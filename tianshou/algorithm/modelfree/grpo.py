import hashlib

import numpy as np
import torch

from tianshou.algorithm.algorithm_base import OnPolicyAlgorithm, TrainingStats
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import ReplayBuffer
from tianshou.data.types import RolloutBatchProtocol



def numpy_hash_rounded(arr: np.ndarray, decimals: int = 6, algorithm='sha1') -> str:
    """Hash numpy array with rounding to handle floating-point precision."""
    h = hashlib.new(algorithm)
    # Round to specified decimals before hashing
    rounded = np.round(arr, decimals=decimals)
    h.update(rounded.tobytes(order='C'))
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

    def _extract_episode_boundaries(self, batch: RolloutBatchProtocol) -> tuple[np.ndarray, np.ndarray]:
        """Extract start and end indices for each episode in the batch.

        Returns:
            start_indices: Array of shape (num_episodes,) starting indices for each episode
            end_indices: Array of shape (num_episodes,) ending indices for each episode
        """
        end_indices = np.where(batch.done)[0]
        start_indices = np.concatenate(([0], end_indices[:-1] + 1))
        return start_indices, end_indices

    def _compute_trajectory_returns(
            self,
            batch: RolloutBatchProtocol,
            start_indices: np.ndarray,
            end_indices: np.ndarray
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
            trajectory_returns[i] = batch.rew[start_idx:end_idx + 1].sum()

        return trajectory_returns

    def _group_trajectories_by_starting_state(
            self,
            starting_states: np.ndarray,
            start_indices: np.ndarray,
            end_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Group trajectories by their starting state using hashing.

        Returns:
            unique_hashes: Array of unique state hashes
            inverse_indices: Mapping from trajectory index to unique hash index
        """
        # Compute hashes for all starting states
        state_hashes = np.array([numpy_hash_rounded(state) for state in starting_states])

        # Group episodes by unique starting state hash
        unique_hashes, inverse_indices = np.unique(state_hashes, return_inverse=True)

        return unique_hashes, inverse_indices

    def _compute_and_assign_advantages_per_state(
            self,
            batch: RolloutBatchProtocol,
            unique_hashes: np.ndarray,
            inverse_indices: np.ndarray,
            starting_states: np.ndarray,
            start_indices: np.ndarray,
            end_indices: np.ndarray,
            trajectory_returns: np.ndarray
    ) -> None:
        """For each unique starting state, compute standardized advantages and assign to batch.

        This method:
        1. Groups trajectories by starting state
        2. Computes mean and std of returns for each group
        3. Calculates standardized advantages: (return - mean) / std
        4. Assigns the advantage to all timesteps in each trajectory
        """
        # Initialize advantage array in batch
        batch.adv = np.zeros(len(batch), dtype=np.float64)
        batch.starting_states = {}

        for i, unique_hash in enumerate(unique_hashes):
            # Find all trajectories with this starting state
            mask = inverse_indices == i
            trajectory_indices_in_group = np.where(mask)[0]

            # Get returns for this group of trajectories
            group_returns = trajectory_returns[mask]

            # Compute mean and std for this starting state group
            mean_return = group_returns.mean()
            std_return = group_returns.std()

            # Avoid division by zero
            if std_return < 1e-8:
                std_return = 1e-8

            # Compute standardized advantages (return - mean) / std
            standardized_advantages = (group_returns - mean_return) / std_return

            # Assign advantages to all timesteps in each trajectory
            episode_indices_list = []
            for traj_idx, advantage in zip(trajectory_indices_in_group, standardized_advantages):
                start_idx = start_indices[traj_idx]
                end_idx = end_indices[traj_idx]

                # Assign the same advantage to all timesteps in this trajectory
                batch.adv[start_idx:end_idx + 1] = advantage

                episode_indices_list.append((start_idx, end_idx))

            # Store metadata for this starting state group
            first_traj_idx = trajectory_indices_in_group[0]
            batch.starting_states[unique_hash] = {
                "state": starting_states[first_traj_idx],
                "indices": episode_indices_list,
                "returns": group_returns,
                "mean_return": mean_return,
                "std_return": std_return,
                "standardized_advantages": standardized_advantages
            }

    def _preprocess_batch(
            self,
            batch: RolloutBatchProtocol,
            buffer: ReplayBuffer,
            indices: np.ndarray,
    ) -> RolloutBatchProtocol:
        """Preprocess batch by computing advantages using GRPO algorithm.

        GRPO groups trajectories by their starting state, computes the mean and std
        of returns within each group, and assigns standardized advantages to each trajectory.
        """
        # Step 1: Extract episode boundaries
        start_indices, end_indices = self._extract_episode_boundaries(batch)

        # Step 2: Extract starting states for all episodes
        starting_states = batch[start_indices].obs

        # Step 3: Compute total return for each trajectory
        trajectory_returns = self._compute_trajectory_returns(batch, start_indices, end_indices)

        # Step 4: Group trajectories by starting state
        unique_hashes, inverse_indices = self._group_trajectories_by_starting_state(
            starting_states, start_indices, end_indices
        )

        # Step 5: Compute and assign advantages for each starting state group
        self._compute_and_assign_advantages_per_state(
            batch,
            unique_hashes,
            inverse_indices,
            starting_states,
            start_indices,
            end_indices,
            trajectory_returns
        )

        logp_old = []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                logp_old.append(self.policy(minibatch).dist.log_prob(minibatch.act))
            batch.logp_old = torch.cat(logp_old, dim=0).flatten()

        return batch

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None, repeat: int
    ) -> TrainingStats:
        """Update policy weights by processing trajectories grouped by initial state.

        For each unique initial state:
        1. Collect ALL trajectories that started from that state into a single batch
        2. Split that batch into minibatches
        3. Process each minibatch and update weights
        4. Move to the next initial state and repeat

        This ensures we fully process all trajectories from one initial state before
        moving to the next, as required by the GRPO algorithm.
        """
        losses, clip_losses, kl_losses = [], [], []
        gradient_steps = 0
        split_batch_size = batch_size or self.max_batchsize

        # Iterate through each unique initial state
        for state_hash, state_info in batch.starting_states.items():
            # Step 1: Get ALL trajectories for this initial state
            # state_info['indices'] contains list of (start_idx, end_idx) for each trajectory
            all_indices_for_this_state = []
            for start_idx, end_idx in state_info['indices']:
                # Collect all timestep indices for this trajectory (inclusive)
                trajectory_indices = list(range(start_idx, end_idx + 1))
                all_indices_for_this_state.extend(trajectory_indices)

            # Create a batch containing all trajectories from this initial state
            state_batch = batch[all_indices_for_this_state]

            # Step 2: Split the state batch into minibatches
            # Step 3: Process each minibatch and update weights
            for minibatch in state_batch.split(split_batch_size, shuffle=False, merge_last=True):
                gradient_steps += 1

                # Calculate loss for actor (sum over all trajectories in minibatch)
                advantages = minibatch.adv
                dist = self.policy(minibatch).dist
                ratios = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                clipped_ratio = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                trajectory_policy_loss = -torch.min(ratios * advantages, clipped_ratio * advantages).mean()
                inv_ratio = 1.0 / ratios
                trajectory_kl_penalty = torch.mean(inv_ratio - torch.log(inv_ratio + 1e-8) - 1)

                loss = trajectory_policy_loss + self.kl_coefficient * trajectory_kl_penalty
                self.optim.step(loss)
                losses.append(loss.item())
                clip_losses.append(trajectory_policy_loss.item())
                kl_losses.append(trajectory_kl_penalty.item())

            # Step 4: All minibatches for this initial state have been processed
            # Weights have been updated, now move to next initial state

        return TrainingStats()