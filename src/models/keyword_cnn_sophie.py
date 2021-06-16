import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyWordCNN1d(nn.Module):
    """
    """

    def __init__(self, num_classes, num_features, num_kernels, mem_depth, task_str='a'):
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
        self.mem_depth = mem_depth
        self.num_kernels = num_kernels
        self.num_classes = num_classes
        self.num_features = num_features
        self.task_str = task_str
        
        if task_str == 'a':
        	print('Task a: one-dimensional convolution layer')
        	self.conv_layer = nn.Conv1d(num_features, num_kernels, kernel_size=mem_depth,
        	padding=mem_depth - 1)  
        elif task_str == 'b':
        	print('Task b: parameter groups=num features')
        	self.conv_layer = nn.Conv1d(num_features, num_kernels, kernel_size=mem_depth,
        	padding=mem_depth - 1, groups = num_features)  
        else: 
        	print('Task ',task_str)
        	self.conv_layer = nn.Conv1d(num_features, num_kernels, kernel_size=mem_depth,
        	padding=mem_depth - 1)  
        	
        self.bn1 = nn.BatchNorm1d(num_kernels)
        self.pool1 = nn.MaxPool1d(30)
        
        self.linear1 = nn.Linear(num_kernels, 128*3)        
        self.linear_out = nn.Linear(128*3, 5)

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

        # output = F.log_softmax(...)
        # return output
        
        if len(x.shape) != 3:
            x = torch.squeeze(x)

        batch_size = x.shape[0]
        in_channels = x.shape[1]  # 40 in our case
        signal_length = x.shape[2]

        x = torch.relu(self.conv_layer(x))

        #print('before pool:',x.shape)
	
        if self.task_str == 'd':
        	x = torch.relu(self.bn1(x))
        elif self.task_str == 'e':
        	#x = torch.relu(self.bn1(x))
        	x = torch.relu(self.pool1(x))

        #print('after pool:',x.shape)
        	
        x = torch.mean(x,dim=2)
        #print('after mean:',x.shape)
        x = torch.relu(self.linear1(x))
        x = self.linear_out(x)
        
        output = F.log_softmax(x,dim=1)
        
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
        	padding=(num_features-1, mem_depth-1))  
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(num_kernels*(2*num_features-1), 128*3)        
        self.linear_out = nn.Linear(128*3, 5)

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
        if len(x.shape) != 4:
            x = torch.unsqueeze(x,0)
            
        batch_size = x.shape[0]
        in_channels = x.shape[1]  # 1 in our case
        num_features = x.shape[2]
        signal_length = x.shape[3]

        x = torch.relu(self.conv_layer(x))	
        x = torch.mean(x,dim=3)
        x = self.flatten(x)
        x = torch.relu(self.linear1(x))
        x = self.linear_out(x)
        
        output = F.log_softmax(x,dim=1)
        
        return output.view(batch_size, -1)[:, :signal_length]
        
