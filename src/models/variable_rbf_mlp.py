import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class VarRbfMLP(nn.Module):
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

    def __init__(self, h_sizes):
        """
        Parameters
        ----------
        h_sizes: list, length of the list specifys the number of hidden layer, value of each entry specifys the dimensionality of the neurons. 
        a_func: defines the activation function
        """
        super().__init__()
        
        self.hidden = nn.ModuleList()
        h_sizes.insert(0, 1) #input layer has dimensionality one!
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.add_module("hidden_layer"+str(k), self.hidden[-1])
            
        self.output_layer = nn.Linear(h_sizes[-1], 1)

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

        # compute forward pass+
        
        # Feedforward
        
        m = nn.Sigmoid()
        #m = nn.Tanh()
        #m = nn.ReLU()
        
        for layer in range(len(self.hidden)-1):
            #x = torch.exp(-torch.pow(self.hidden[layer](x),2))
            x = m(self.hidden[layer](x))
            
        output = self.output_layer(x)
        return output
    
    def predict(self, x):
        """Compute the network output for a given input and return the result as numpy array.

        Parameters
        ----------
        x : float or numpy.array or torch.Tensor
            The network input

        Returns
        ----------
        numpy.array
            The network output in (batch_size, 1) shape
        """
        output = self(x)
        return output.detach().numpy()    
    
    
    def fit(self, data, targets, batch_size=5, learning_rate=1e-2, max_epochs=300):
        """Train the network on the given data.

        Parameters
        ----------
        data : numpy.array
            The input training samples
        targets : numpy.array
            The output training samples
        batch_size : int
            Batch size for mini-batch gradient descent training
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

        # create a data loader that takes care of batching our training set
        data = torch.FloatTensor(data[:, np.newaxis]) if len(data.shape) < 2 else torch.FloatTensor(data)
        targets = torch.FloatTensor(targets[:, np.newaxis]) if len(targets.shape) < 2 else torch.FloatTensor(targets)
        training_set = TensorDataset(data, targets)
        train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

        # train the network
        train_loss_list = []
        for i in range(max_epochs):
            for batch in train_loader:
                batch_data = batch[0]
                batch_targets = batch[1]

                # compute batch loss
                predictions = self(batch_data)
                batch_loss = loss(predictions, batch_targets)

                # optimize the network parameters
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            # record the training set MSE after each epoch; `with torch.no_grad()` makes sure that we do not record
            # gradient information of anything inside this environment
            with torch.no_grad():
                predictions = self(training_set[:][0])
                train_loss = loss(predictions, training_set[:][1])
                train_loss_list.append(train_loss.detach().numpy())

            # log the training loss
            print(f'Epoch {i} training loss is {train_loss:.2f}', end='\r')
            if i % 50 == 0:
                print('')

        return train_loss_list

  # https://www.youtube.com/watch?v=9j-_dOze4IM