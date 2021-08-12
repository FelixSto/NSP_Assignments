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
        self.feat_conv = nn.Conv1d(num_channels, num_channels, kernel_size, dilation=self.dilation)
        self.gate_conv = nn.Conv1d(num_channels, num_channels, kernel_size, dilation=self.dilation)
        self.mix_conv =  nn.Conv1d(num_channels, num_channels, kernel_size=1)
        
        #print('been in residual init')

    def buffer_append(self, x: torch.Tensor):
        """ Append a sample to the convolution buffer.

        Parameters
        ----------
        x : torch.Tensor
            Sample to append to the convolution buffer.
        """
        #print('been here 1')
        assert x.shape == (1, self.num_channels, 1)
        self.conv_buffer = torch.cat((self.conv_buffer[:, :, 1:], x), dim=-1)
        #print('been in residual buffer append')
        

    def generate(self):
        """ Generate a single sample by evaluating the convolution buffer.

        Returns
        ----------
        torch.Tensor, torch.Tensor
            The skip and residual sample of shape (1, num_channels, 1).
        """

        # ToDo: compute a forward pass given the conv buffer       
        
        x = self.conv_buffer
        
        pad_length = round(self.dilation*(self.kernel_size-1))
        
        x_pad = F.pad(x,(pad_length, 0))
        
        z = torch.tanh(self.feat_conv(x_pad)) * torch.sigmoid(self.gate_conv(x_pad))
        z = self.mix_conv(z[:,:,-1].unsqueeze(dim=2))
        
        skip = z
        residual= (skip + x[:,:,-1].unsqueeze(dim=2))
        
        assert skip.shape == (1, self.num_channels, 1)
        assert residual.shape == (1, self.num_channels, 1)
                
        return skip, residual

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
        
        pad_length = round(self.dilation*(self.kernel_size-1))
        
        x_pad = F.pad(x,(pad_length, 0))

        z = torch.tanh(self.feat_conv(x_pad)) * torch.sigmoid(self.gate_conv(x_pad))
        
        if fill_buffer:
            self.conv_buffer = z[:, :, -self.buffer_size:]
            assert self.conv_buffer.shape == (1, self.num_channels, self.buffer_size)
            
        z = self.mix_conv(z)
        skip = z
        residual= z + x
        
        
        assert skip.shape == (batch_size, self.num_channels, signal_length)
        assert residual.shape == (batch_size, self.num_channels, signal_length)

        return skip, residual
