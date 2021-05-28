import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A simple convolutional network with a single conv layer for one dimensional, single channel input signals, a
    a fully connected hidden and output layer.

    Attributes
    ----------
    num_kernels : int
        Number of convolution kernels (default is 3)
    mem_depth : int
        Memory depth = kernel size of the model (default is 5)
    """

    def __init__(self, num_kernels=3, mem_depth=5):
        """
        Parameters
        ----------
        num_kernels : int
            Number of convolution kernels (default is 3)
        mem_depth : int
            Memory depth = kernel size of the model (default is 5)
        """
        super().__init__()
        self.mem_depth = mem_depth
        self.num_kernels = num_kernels
        self.conv_layer = nn.Conv1d(1, num_kernels, kernel_size=mem_depth, padding=mem_depth - 1)  # use zero-padding
        self.linear = nn.Linear(num_kernels, 32)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        """Forward pass of the network.

        Parameters
        ----------
        x : float or numpy.array or torch.Tensor
            The network input

        Returns
        ----------
        torch.Tensor
            The network output in (batch_size, signal_length) shape
        """
        # convert input to torch.Tensor if necessary
        if not torch.is_tensor(x):
            x = torch.FloatTensor(x)

        # get input into shape (batch_size=1, in_channels=1, signal_length) if no proper shape given
        if len(x.shape) != 3:
            x = x.view((1, 1, -1))

        batch_size = x.shape[0]
        in_channels = x.shape[1]  # 1 in our case
        signal_length = x.shape[2]

        # compute forward pass
        z1 = torch.relu(self.conv_layer(x))  # z1 has shape (batch_size, out_channels, signal_length)
        z2 = z1.permute(0, 2, 1)  # permute axis such that we can combine the different conv channels with a fc layer
        z3 = torch.relu(self.linear(z2))
        output = self.output_layer(z3)
        return output.view(batch_size, -1)[:, :signal_length]  # cut-off the invalid samples

    def predict(self, x):
        """Compute the network output for a given input and return the result as numpy array.

        Parameters
        ----------
        x : float or numpy.array or torch.Tensor
            The network input

        Returns
        ----------
        numpy.array
            The network output in (batch_size, signal_length) shape
        """
        output = self(x)
        return output.detach().numpy()

    def fit(self, data, targets, learning_rate=5e-3, max_epochs=300):
        """Train the network on the given data.

        Parameters
        ----------
        data : numpy.array
            The input training signal(s)
        targets : numpy.array
            The output training signal(s)
        learning_rate : float
            The learning rate for the optimizer
        max_epochs : int
            Maximum number of training epochs

        Returns
        ----------
        list
            List of training set MSE scores for each training epoch
        """
        # define the loss function (MSE) and the optimizer (Adam)
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # make targets a tensor
        targets = torch.FloatTensor(targets)

        # train the network
        train_loss_list = []
        for i in range(max_epochs):
            # compute training loss
            predictions = self(data)
            train_loss = loss(predictions, targets)

            # optimize the network parameters
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # record the training set MSE after each epoch
            train_loss_list.append(train_loss.detach().numpy())

            # log the training loss
            print(f'Epoch {i} training loss is {train_loss:.2f}', end='\r')
            if i % 50 == 0:
                print('')

        return train_loss_list

    def save(self, path):
        """Save the network weights

        Parameters
        ----------
        path : str
            Target storage path
        """
        torch.save({
            'state_dict': self.state_dict(),
        }, path)

    def load(self, path):
        """Load the network weights from a file

        Parameters
        ----------
        path : str
            File containing the stored weights
        """
        model = torch.load(path)
        self.load_state_dict(model['state_dict'])
