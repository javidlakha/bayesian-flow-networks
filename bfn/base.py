from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torchtyping import TensorType


class BFN(nn.Module, ABC):
    """Base class for a Bayesian Flow Network."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def output_distribution(
        self,
        input_distribution: TensorType['batch', 'sequence', 'vocab'],
        t: TensorType['batch'],
    ) -> TensorType['batch', 'sequence', 'vocab']:
        """
        Computes the joint probability distribution of a sequence, given the
        probability distributions of each element in the sequence and the
        normalised timestep.
        """
        pass

    @abstractmethod
    def continuous_time_loss(
        self,
        x: TensorType['batch', 'sequence'],
    ) -> torch.float32:
        """Computes the continuous time loss for a batch of training data."""
        pass

    @abstractmethod
    def generate(
        self,
        batch_size: int,
        steps: int,
        device: torch.device,
    ) -> TensorType['batch', 'sequence']:
        """Generates a batch of data."""
        pass
