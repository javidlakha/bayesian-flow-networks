from typing import Optional

import torch
import torch.nn.functional as F
from torchtyping import TensorType

from bfn.base import BFN


class DiscreteBFN(BFN):
    """
    Implements a discrete Bayesian Flow Network.

    Based on Alex Graves, Rupesh Kumar Srivastava, Timothy Atkinson and
    Faustino Gomez (2023), 'Bayesian Flow Networks',
    https://arxiv.org/abs/2308.07037
    """

    def __init__(
        self,
        sequence_len: int,
        vocab_size: int,
        net: torch.nn,
        beta: float = 3.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.net = net

    def output_distribution(
        self,
        input_distribution: TensorType['batch', 'sequence', 'vocab'],
        t: TensorType['batch'],
        context: Optional[TensorType['batch', 'context']] = None,
    ) -> TensorType['batch', 'sequence', 'vocab']:
        """
        Computes the joint probability distribution of a sequence, given the
        probability distributions of each element in the sequence and the
        normalised timestep.

        Given
          - the input distribution of multinomial class probabilities for each
            element in the sequence, computed from the input data based on the
            assumption that each element is independent

          - the normalised time step, `t`, which determines how noisy the input
            distribution is

        a neural network is used to compute the multinomial class probabilities
        jointly for each element in the sequence (i.e. without assuming
        independence).

        Based on the Discrete Output Distribution subroutine of Algorithm 8 in
        the Bayesian Flow Networks paper.
        """
        output_distribution = self.net(input_distribution, t, context=context)
        return F.softmax(output_distribution, dim=-1)

    def continuous_time_loss(
        self,
        x: TensorType['batch', 'sequence'],
        context: Optional[TensorType['batch', 'context']] = None,
    ) -> torch.float32:
        """
        Computes the continuous time loss for a batch of training data.

        Given `x`, a sequence of discrete data:
          - samples a timestep uniformly at random

          - samples the multinomial class probabilities that might have been
            used to generate each element of `x`, assuming each element is
            independent

          - computes the multinomial class probabilities that might have been
            used to generate each element of `x`, jointly for the sequence

          - computes the loss between the predicted class probabilities and
            the ground truth.

        Based on Algorithm 8 in the Bayesian Flow Networks paper.
        """
        batch_size, sequence_len = x.shape
        assert sequence_len == self.sequence_len

        # Sample t (the normalised time step) uniformly at random from the
        # interval [0, 1) for each element in the batch
        t = torch.rand(batch_size, device=x.device, dtype=torch.float32)

        # Approximate the accuracy schedule β(t) using the rule of thumb
        #
        #   β(t) = t^2 * β(1)
        #
        # (The derivation of the accuracy schedule is outlined in Section 6.8)
        accuracy = t**2 * self.beta

        # At each sequence position (for each observation in the batch), sample
        #
        #   y ~ β(t) * Normal(vocab_size * OneHot(x) - 1, vocab_size * I)
        #
        # This is used to model the 'input distribution' - i.e. the multinomial
        # class probabilities at each sequence position, computed on the basis
        # that each element in the sequence is independent
        x = F.one_hot(x, num_classes=self.vocab_size).float()
        mean = accuracy[:, None, None] * (self.vocab_size * x - 1)
        std_dev = (accuracy * self.vocab_size)[:, None, None].sqrt()
        y = mean + std_dev * torch.randn_like(mean)
        input_distribution = F.softmax(y, dim=-1)

        # Model the 'output distribution' - i.e. the joint multinomial class
        # probabilities for the sequence, computed without assuming that each
        # element is independent
        output_distribution = self.output_distribution(
            input_distribution, t, context
        )

        # Compute the loss between the output distribution and the ground truth
        # (The derivation of the loss function is outlined in Sections 6.10 -
        # 6.12)
        loss = (
            self.vocab_size * self.beta * t[:, None, None] *
            (x - output_distribution)**2
        )
        return loss.mean()

    @torch.inference_mode()
    def generate(
        self,
        batch_size: int,
        steps: int,
        device: torch.device,
        context: Optional[TensorType['batch', 'context']] = None,
    ) -> TensorType['batch', 'sequence']:
        """
        Generates a batch of data.

        Implements Algorithm 9 in the Bayesian Flow Networks paper.
        """
        previous_mode = self.net.training
        self.net.eval()

        # Use a uniform prior. The network will learn the empirical prior of
        # the training set and use that to correct its predictions
        prior = (1/self.vocab_size) * torch.ones((
            batch_size, self.sequence_len, self.vocab_size
        ), device=device)

        for step in range(1, steps):
            # Compute the joint probability distribution for each element in
            # the sequence. This is done using a learned neural network that
            # corrects the prior class probabilities, which were computed based
            # on the (unrealistic) assumption that each element in the sequence
            # is independent
            t = (step/steps) * torch.ones(batch_size, device=device)
            element_probs = self.output_distribution(prior, t, context)

            # Sample from the computed distribution and use this to update the
            # prior class probabilities for each element in the sequence. This
            # is done based on the assumption that each element in the sequence
            # is independent. The update equations are derived in Section 6.5
            # of the Bayesian Flow Networks paper
            sample = torch.distributions.Categorical(element_probs).sample()
            sample = F.one_hot(sample, num_classes=self.vocab_size).float()
            alpha = self.beta * (2*(step+1)-1)/steps**2
            mean = alpha * (self.vocab_size * sample - 1)
            std_dev = torch.full_like(
                mean,
                fill_value=(alpha*self.vocab_size),
            ).sqrt()
            y = mean + std_dev * torch.randn_like(sample)
            posterior = torch.exp(y) * prior
            posterior = posterior / posterior.sum(-1, keepdim=True)

            # Use the posterior distribution as the prior for the next step
            prior = posterior

        t = torch.ones(batch_size, device=device)
        element_probs = self.output_distribution(prior, t, context)
        sample = torch.distributions.Categorical(element_probs).sample()

        self.net.training = previous_mode

        return sample
