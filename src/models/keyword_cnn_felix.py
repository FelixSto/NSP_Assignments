import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyWordCNN1d(nn.Module):
    """
    """

    def __init__(self, num_classes, num_features, num_kernels ,mem_depth, num_groups):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes to predict
        num_features : int
            Number of signal features per time step
        num_kernels : int
            Number of convolution kernels
        mem_depth : int
            Memory depth = kernel size of the model
        """
        super().__init__()
        self.num_kernels = num_kernels
        self.mem_depth = mem_depth
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_groups = num_groups
        
        #TODO: define your model here
                
        self.conv_layer = nn.Conv1d(num_features, num_kernels, kernel_size=mem_depth, 
                                    groups = num_groups, padding=mem_depth - 1)  
        self.linear = nn.Linear(num_kernels, 256)
        self.output_layer = nn.Linear(256, 5)

    def forward(self, x:torch.Tensor):
        """Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The network input of shape (batch_size, 1, in_channels, sequence_length)

        Returns
        ----------
        torch.Tensor
            The network output (softmax logits) of shape (batch_size, num_classes)
        """        
        
        # TODO: implement the forward pass here
        
        if len(x.shape) != 3:
            x = torch.squeeze(x)
            
        batch_size = x.shape[0]    # depends
        in_channels = x.shape[1]   # 40 
        signal_length = x.shape[2] # 81

        x = torch.relu(self.conv_layer(x))
        x = torch.mean(x,dim=2)
        #print('Post conv shape: ', x.shape) [256,20]
        x = torch.relu(self.linear(x))
        x = self.output_layer(x)
        output = F.log_softmax(x,dim=1)
        
        # return output
        
        return output.view(batch_size, -1)[:, :signal_length]


class KeyWordCNN2d(nn.Module):
    """
    """

    def __init__(self, num_classes, num_features, num_kernels, mem_depth):
        
        """
        Parameters
        ----------
        num_classes : int
            Number of classes to predict
        num_features : int
            Number of signal features per time step
        num_kernels : int
            Number of convolution kernels
        mem_depth : int
            Memory depth = kernel size of the model
        """
        super().__init__()
        self.num_features = num_features
        self.num_kernels = num_kernels
        self.mem_depth = mem_depth
        self.num_classes = num_classes
        #TODO: define your model here
        
        self.conv_layer = nn.Conv2d(1, num_kernels, kernel_size=(num_features, mem_depth), 
                                    padding=(num_features-1,mem_depth - 1))
        
        self.linear = nn.Linear(num_kernels*(2*num_features-1), 256)
        self.output_layer = nn.Linear(256, 5)

    def forward(self, x):
        """Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The network input of shape (batch_size, in_channels, num_features, sequence_length)

        Returns
        ----------
        torch.Tensor
            The network output (softmax logits) of shape (batch_size, num_classes)
        """
          
        #TODO: implement the forward pass here
        
        if len(x.shape) != 4:
            x = torch.unsqueeze(x,0)
        
        #print('Input shape: ', x.shape)
            
        batch_size = x.shape[0]   # depends
        in_channels = x.shape[1]  # 40 
        signal_length = x.shape[3]# Variable

        x = torch.relu(self.conv_layer(x))
        x = torch.mean(x,dim=3)
        #print('Post conv shape 1: ', x.shape)
        
        x = torch.flatten(x, start_dim=1)
        
        #print('Post conv shape 2: ', x.shape)
        x = torch.relu(self.linear(x))
        x = self.output_layer(x)
        output = F.log_softmax(x,dim=1)
        
        # return output
        
        return output.view(batch_size, -1)[:, :signal_length]
