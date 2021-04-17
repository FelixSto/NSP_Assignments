import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class RbfMLP(nn.Module):
    """
    A simple feed-forward network with one input/output unit, two hidden layers and tanh activation.

    Attributes
    ----------
    hidden_size : int
        Number of hidden units per hidden layer
    hidden_layer_1 : str
        the name of the animal
    hidden_layer_2 : str
        the sound that the animal makes
    output_layer : int
        the number of legs the animal has (default 4)
    """

    def __init__(self, hidden_size=15):
        """
        Parameters
        ----------
        hidden_size : int
            Number of hidden units per hidden layer (default is 32)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_layer_1 = nn.Linear(1, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        """Forward pass of the network.

        Parameters
        ----------
        x : float or numpy.array or torch.Tensor
            The network input

        Returns
        ----------
        torch.Tensor
            The network output in (batch_size, 1) shape
        """
        # convert input to torch.Tensor if necessary
        if not torch.is_tensor(x):
            x = torch.FloatTensor(x)

        # get input into shape (batch_size, 1)
        x = x.view((-1, 1))

        # compute forward pass
        z1 = torch.exp(torch.pow(self.hidden_layer_1(x),2))
        output = self.output_layer(z1)
        return output

  # https://www.youtube.com/watch?v=9j-_dOze4IM