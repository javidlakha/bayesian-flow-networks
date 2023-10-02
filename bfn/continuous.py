import math
from typing import Optional

import torch
from torchtyping import TensorType

from bfn.base import BFN


class ContinuousBFN(BFN):
    """
    Implements a continuous Bayesian Flow Network.

    Based on Alex Graves, Rupesh Kumar Srivastava, Timothy Atkinson and
    Faustino Gomez (2023), 'Bayesian Flow Networks',
    https://arxiv.org/abs/2308.07037
    """

    def __init__(
        self,
        channels: int,
        sequence_len: int,
        net: torch.nn,
        sigma: float = 0.001,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.sequence_len = sequence_len
        self.net = net
        self.sigma = sigma

    def output_distribution(
        self,
        input_distribution: TensorType['batch', 'channel', 'sequence'],
        t: TensorType['batch'],
        accuracy: TensorType['batch'],
        context: Optional[TensorType['batch', 'context']] = None,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> TensorType['batch', 'channel', 'sequence']:
        """
        Computes the joint probability distribution of a sequence, given the
        probability distributions of each element in the sequence, the accuracy
        and the normalised timestep.

        Given
          - the input distribution of multinomial class probabilities for each
            element in the sequence, computed from the input data based on the
            assumption that each element is independent

          - the normalised time step, `t`, and the accuracy schedule, which
            determine how noisy the input distribution is

        a neural network is used to compute the multinomial class probabilities
        jointly for each element in the sequence (i.e. without assuming
        independence).

        Based on the Continuous Output Prediction subroutine of Algorithm 2 in
        the Bayesian Flow Networks paper.
        """
        output_distribution = (
            (input_distribution/accuracy[:, None, None]) -
            ((1-accuracy)/accuracy)[:, None, None].sqrt() *
            self.net(input_distribution, t, context=context)
        )

        if minimum or maximum:
            output_distribution = output_distribution.clamp(minimum, maximum)

        return output_distribution

    def continuous_time_loss(
        self,
        x: TensorType['batch', 'channel', 'sequence'],
        context: Optional[TensorType['batch', 'context']] = None,
        t_min: float = 1e-6,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> torch.float32:
        """
        Computes the continuous time loss for a batch of training data.

        Given `x`, a sequence of continuous data:
          - samples a timestep uniformly at random

          - samples the mean and variance that might have been used to generate
            each element of `x`, assuming the data are normally distributed

          - computes the mean and variance that might have been used to
            generate each element of `x`, jointly for the sequence

          - computes the loss between the predicted class probabilities and
            the ground truth.

        Based on Algorithm 2 in the Bayesian Flow Networks paper.
        """
        batch_size, channels, sequence_len = x.shape
        assert channels == self.channels
        assert sequence_len == self.sequence_len

        # Sample t (the normalised time step) uniformly at random from the
        # interval [`t_min`, 1) for each element in the batch
        t = torch.rand(batch_size, device=x.device, dtype=torch.float32)
        t = (t + t_min) * (1 - t_min)

        # Compute the accuracy schedule γ(t) using the formula
        #
        #   γ(t) = 1 - σ(1) ** 2t
        #
        # (The derivation of the accuracy schedule is outlined in Sections 4.5
        # and 4.6)
        accuracy = 1 - self.sigma ** (2*t)

        # At each sequence position (for each observation in the batch), sample
        #
        #   y ~ Normal(γ(t) * x, γ(t) * (1 - γ(t)) * I)
        #
        # This is used to model the 'input distribution' - i.e. the mean and
        # variance at each sequence position, computed on the basis that each
        # element in the sequence is independent
        mean = accuracy[:, None, None] * x
        std_dev = (accuracy * (1 - accuracy)).sqrt()[:, None, None]
        input_distribution = mean + std_dev * torch.randn_like(mean)

        # Model the 'output distribution' - i.e. the joint means and variances
        # for the sequence, computed without assuming that each element is
        # independent
        output_distribution = self.output_distribution(
            input_distribution, t, accuracy, context, minimum, maximum
        )

        # Compute the loss between the output distribution and the ground truth
        # (The derivation of the loss function is outlined in Sections 4.10 -
        # 4.12)
        loss = (
            (-math.log(self.sigma) * (self.sigma ** (-2 * t)))[:, None, None] *
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
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> TensorType['batch', 'sequence']:
        """
        Generates a batch of data.

        Implements Algorithm 3 in the Bayesian Flow Networks paper.
        """
        previous_mode = self.net.training
        self.net.eval()

        # Use a standard normal distribution as the prior. The network will
        # learn the empirical prior of the training set and use that to correct
        # its predictions
        dimensions = (batch_size, self.channels, self.sequence_len)
        mean = torch.zeros(dimensions, device=device)
        variance = torch.ones(dimensions, device=device)

        for step in range(1, steps):
            # Compute the joint probability distribution for each element in
            # the sequence. This is done using a learned neural network that
            # corrects the prior class probabilities, which were computed based
            # on the (unrealistic) assumption that each element in the sequence
            # is independent
            t = (step/steps) * torch.ones(batch_size, device=device)
            accuracy = 1 - self.sigma ** (2*t)
            output_distribution = self.output_distribution(
                mean,
                t,
                accuracy,
                context,
                minimum,
                maximum,
            )

            # Sample from the computed distribution and use this to update the
            # prior class probabilities for each element in the sequence. This
            # is done based on the assumption that each element in the sequence
            # is independent. The update equations are derived in Section 4.2
            # of the Bayesian Flow Networks paper
            precision = self.sigma**(-2*step/steps) * (1-self.sigma**(2/steps))
            sample = (
                output_distribution +
                (1/precision)**(1/2) * torch.randn_like(output_distribution)
            )
            mean = (
                (variance * mean + precision * sample) /
                (variance + precision)
            )
            variance = variance + precision

        t = torch.ones(batch_size, device=device)
        accuracy = 1 - self.sigma ** (2*t)
        sample = self.output_distribution(
            mean,
            t,
            accuracy,
            context,
            minimum,
            maximum,
        )

        self.net.training = previous_mode

        return sample
