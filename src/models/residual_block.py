import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A residual block as presented in the WaveNet paper.
    """

    def __init__(self, num_channels, kernel_size, dilation):
        """
        Parameters
        ----------
        num_channels : int
            Number of input and output channels (convolution kernels)
        kernel_size : int
            Size of the convolution kernels
        dilation : int
            Dilation of the convolution layers.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.dilation = dilation

        self.buffer_size = dilation * (kernel_size - 1) + 1
        self.conv_buffer = torch.zeros(1, num_channels, self.buffer_size)

        # ToDo: define your conv layers
        # self.feat_conv =
        # self.gate_conv =
        # self.mix_conv =

    def buffer_append(self, x: torch.Tensor):
        """ Append a sample to the convolution buffer.

        Parameters
        ----------
        x : torch.Tensor
            Sample to append to the convolution buffer.
        """
        assert x.shape == (1, self.num_channels, 1)
        self.conv_buffer = torch.cat((self.conv_buffer[:, :, 1:], x), dim=-1)

    def generate(self):
        """ Generate a single sample by evaluating the convolution buffer.

        Returns
        ----------
        torch.Tensor, torch.Tensor
            The skip and residual sample of shape (1, num_channels, 1).
        """

        # ToDo: compute a forward pass given the conv buffer

        # assert skip.shape == (1, self.num_channels, 1)
        # assert residual.shape == (1, self.num_channels, 1)
        #
        # return skip, residual

    def forward(self, x, fill_buffer=False):
        """Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The network input of shape (batch_size, num_channels, signal_length)
        fill_buffer : bool
            If true, fill the convolution buffer of the block

        Returns
        ----------
        torch.Tensor, torch.Tensor
            The skip and residual connections of shape (batch_size, num_channels, signal_length)
        """
        assert torch.is_tensor(x) and len(x.shape) == 3

        batch_size = x.shape[0]
        signal_length = x.shape[2]

        # ToDo: do proper padding

        if fill_buffer:
            self.conv_buffer = z[:, :, -self.buffer_size:]
            assert self.conv_buffer.shape == (1, self.num_channels, self.buffer_size)


        # ToDo: implement the forward pass

        assert skip.shape == (batch_size, self.num_channels, signal_length)
        assert residual.shape == (batch_size, self.num_channels, signal_length)

        return skip, residual
