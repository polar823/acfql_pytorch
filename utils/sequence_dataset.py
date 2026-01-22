"""PyTorch Dataset wrapper for sequence sampling.

This module provides a PyTorch-compatible Dataset wrapper that enables
efficient multi-process data loading with DataLoader.
"""

import torch
import numpy as np
from typing import Dict, Any


class SequenceDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for sequence sampling.

    Wraps the custom Dataset class to enable use with PyTorch DataLoader
    for multi-process data loading and prefetching.

    Args:
        dataset: The offline RL dataset
        sequence_length: Length of action sequences to sample
        discount: Discount factor for computing returns
    """

    def __init__(self, dataset, sequence_length: int, discount: float = 0.99):
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.discount = discount

        # Pre-compute valid sampling indices (avoid episode boundaries)
        self._compute_valid_indices()

    def _compute_valid_indices(self):
        """Compute indices where we can safely sample sequences."""
        # Get episode boundaries
        terminal_indices = self.dataset.terminal_indices
        episode_starts = self.dataset.episode_starts

        valid_indices = []
        for start_idx in episode_starts:
            # Find the end of this episode
            end_idx = terminal_indices[terminal_indices >= start_idx][0]

            # Can sample from start_idx to (end_idx - sequence_length + 1)
            episode_valid = list(range(start_idx, max(start_idx, end_idx - self.sequence_length + 2)))
            valid_indices.extend(episode_valid)

        self.valid_indices = np.array(valid_indices)

    def __len__(self):
        """Return number of valid sampling positions."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Sample a sequence starting from the given index.

        Args:
            idx: Index into valid_indices

        Returns:
            Dictionary containing observations, actions, etc.
        """
        # Map to actual dataset index
        start_idx = self.valid_indices[idx]

        # Sample sequence using the dataset's method
        # Note: We sample batch_size=1 and then squeeze
        batch = self.dataset.sample_sequence(
            batch_size=1,
            sequence_length=self.sequence_length,
            discount=self.discount,
            start_indices=np.array([start_idx])
        )

        # Remove batch dimension (squeeze first dim)
        result = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                # Handle nested dicts (for image observations)
                result[key] = {k: v[0] for k, v in value.items()}
            else:
                result[key] = value[0]

        return result


class RandomSequenceDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for random sequence sampling (simpler version).

    This version doesn't pre-compute indices, just samples randomly
    each time. Simpler but may sample across episode boundaries.

    Args:
        dataset: The offline RL dataset
        sequence_length: Length of action sequences to sample
        discount: Discount factor
        size: Virtual size of dataset (for DataLoader)
    """

    def __init__(self, dataset, sequence_length: int, discount: float = 0.99, size: int = 10000):
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.discount = discount
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Sample a random sequence.

        Args:
            idx: Ignored (we sample randomly)

        Returns:
            Dictionary containing observations, actions, etc.
        """
        # Sample sequence using the dataset's method
        batch = self.dataset.sample_sequence(
            batch_size=1,
            sequence_length=self.sequence_length,
            discount=self.discount
        )

        # Remove batch dimension
        result = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                # Handle nested dicts (for image observations)
                result[key] = {k: v[0] for k, v in value.items()}
            else:
                result[key] = value[0]

        return result
