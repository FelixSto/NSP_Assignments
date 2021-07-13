import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.residual_block import ResidualBlock


class WaveNetDeterministic(nn.Module):
    """
    A modified version of WaveNet with a deterministic output layer.
    """

    def __init__(self, blocks_per_cell, num_cells, num_kernels, kernel_size=2, dilation_factor=2,
                 in_channels=1, hidden_size=128):
        """
        Parameters
        ----------
        blocks_per_cell : int
            Number of residual blocks per WaveNet cell
        num_cells : int
            Number of WaveNet cells
        num_kernels : int
            Number of kernels in the residual convolutions
        kernel_size : int
            Kernel size of the (residual) convolutions
        dilation_factor : int
            Dilation factor. The dilation within a WaveNet cell grows exponentially with the dilation factor as base.
        in_channels : int
            Number of input channels for the inital convolution layer.
        hidden_size : int
            Number of hidden neurons (= number of kernels) of the 1x1 convolution before the output layer.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.in_channels = in_channels
        self.dilation_factor = dilation_factor
        self.residual_blocks_per_cell = blocks_per_cell
        self.receptive_field = kernel_size + (kernel_size - 1) * dilation_factor * (
                dilation_factor ** blocks_per_cell - 1) // (dilation_factor - 1)

        self.initial_conv = nn.Conv1d(in_channels, num_kernels, kernel_size)
        res_blocks = []
        for m in range(num_cells):
            for k in range(1, blocks_per_cell):
                dilation = dilation_factor ** k
                res_blocks.append(ResidualBlock(num_kernels, kernel_size, dilation))

        self.residual_blocks = nn.ModuleList(res_blocks)
        self.skip_conv = nn.Conv1d(num_kernels, hidden_size, kernel_size=1)
        self.out_conv = nn.Conv1d(hidden_size, 1, kernel_size=1)

    def forward(self, x, fill_buffers=False):
        """Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The network input of shape (batch_size, in_channels, signal_length)
        fill_buffers : bool
            If true, fill the conv buffers of the residual blocks during the forward pass

        Returns
        ----------
        torch.Tensor
            The network output of shape (batch_size, 1, signal_length)
        """
        # convert input to torch.Tensor if necessary
        assert torch.is_tensor(x) and len(x.shape) == 3 and x.shape[1] == self.in_channels

        batch_size = x.shape[0]
        signal_length = x.shape[2]

        x = self.initial_conv(F.pad(x, (self.kernel_size - 1, 0), value=0.))
        assert x.shape == (batch_size, self.num_kernels, signal_length)

        skip_sum = torch.zeros(batch_size, self.num_kernels, signal_length)
        for residual_block in self.residual_blocks:
            skip, x = residual_block(x, fill_buffers)
            skip = F.pad(skip, (signal_length - skip.shape[-1], 0), value=0.)
            skip_sum += skip

        z = torch.relu(skip_sum)
        z = torch.relu(self.skip_conv(z))
        z = self.out_conv(z)
        assert z.shape == (batch_size, 1, signal_length)
        return z

    def generate(self, num_samples, initial_samples=None, include_first_samples=False):
        """Generate a signal.

        Parameters
        ----------
        num_samples : int
            The number of samples to generate.
        initial_samples : torch.Tensor
            Samples to initialize the network for generation.
        include_first_samples : bool
            If true, include the initial samples in the returned signal.

        Returns
        ----------
        torch.Tensor
            The generated signal of of shape (num_samples)
        """
        self.eval()
        assert initial_samples is None or len(initial_samples.shape) == 3

        # generate initial samples if none are given
        if initial_samples is None:
            initial_samples = torch.rand(1, 1, self.receptive_field)

        # create and fill conv buffer of first layer
        conv_buffer = torch.zeros(1, self.in_channels, self.kernel_size)
        num_initial = initial_samples.shape[-1]
        if num_initial >= self.kernel_size:
            conv_buffer = initial_samples[:, :, -self.kernel_size:]
        else:
            conv_buffer[:, :, -num_initial:] = initial_samples

        # generate first sample and fill conv buffers of the residual blocks
        first_sample = self(initial_samples, fill_buffers=True)[:, :, -1].unsqueeze(2)
        conv_buffer = torch.cat((conv_buffer[:, :, 1:], first_sample), dim=-1)

        # create and init signal buffer
        signal = []
        if include_first_samples:
            signal.extend(initial_samples.squeeze())
        signal.append(first_sample.squeeze())

        # generate samples
        for i in range(1, num_samples):
            residual = self.initial_conv(conv_buffer)

            skip_sum = torch.zeros(1, self.num_kernels, 1)
            for residual_block in self.residual_blocks:
                residual_block.buffer_append(residual)
                skip, residual = residual_block.generate()
                skip_sum += skip

            z = torch.relu(skip_sum)
            z = torch.relu(self.skip_conv(z))
            next_sample = self.out_conv(z)
            conv_buffer = torch.cat((conv_buffer[:, :, 1:], next_sample), dim=-1)
            signal.append(next_sample.squeeze())

        return torch.stack(signal).squeeze()

    def generate_slow(self, num_samples, initial_samples=None, include_first_samples=False):
        """Generate a signal..

        Parameters
        ----------
        num_samples : int
            The number of samples to generate.
        context : torch.Tensor
            The context embedding vector.
        initial_samples : torch.Tensor
            Samples to initialize the network for generation.
        include_first_samples : bool
            If true, include the initial samples in the returned signal.

        Returns
        ----------
        torch.Tensor
            The generated signal of of shape (num_samples)
        """
        self.eval()
        assert initial_samples is None or len(initial_samples.shape) == 3

        # generate initial samples if none are given
        if initial_samples is None:
            initial_samples = torch.rand(1, 1, self.receptive_field)

        # create and fill conv buffer of first layer
        conv_buffer = torch.zeros(1, self.in_channels, self.receptive_field)
        num_initial = initial_samples.shape[-1]
        if num_initial >= self.receptive_field:
            conv_buffer = initial_samples[:, :, -self.receptive_field:]
        else:
            conv_buffer[:, :, -num_initial:] = initial_samples

        # create and init signal buffer
        signal = []
        if include_first_samples:
            signal.extend(initial_samples.squeeze())

        # generate samples
        for i in range(0, num_samples):
            x = self(conv_buffer)
            next_sample = x[:, :, -1].unsqueeze(2)
            conv_buffer = torch.cat((conv_buffer[:, :, 1:], next_sample), dim=-1)
            signal.append(next_sample.squeeze())

        return torch.stack(signal).squeeze()
