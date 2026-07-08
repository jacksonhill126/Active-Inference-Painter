from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class TransitionBatch:
    state: torch.Tensor
    action: torch.Tensor
    next_state: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.data: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.data)

    def add(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> None:
        self.data.append((state.astype(np.float32), action.astype(np.float32), next_state.astype(np.float32)))

    def sample(self, batch_size: int, device: torch.device) -> TransitionBatch:
        indices = self.rng.integers(0, len(self.data), size=batch_size)
        states, actions, next_states = zip(*(self.data[i] for i in indices))
        return TransitionBatch(
            torch.tensor(np.stack(states), device=device),
            torch.tensor(np.stack(actions), device=device),
            torch.tensor(np.stack(next_states), device=device),
        )
