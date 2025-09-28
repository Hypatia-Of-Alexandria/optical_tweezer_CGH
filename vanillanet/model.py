import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'VanillaModel'
]

class VanillaModel(nn.Module):
    def __init__(self, size, max_points, h1, h2): 
        super().__init__() 
        self.max_points = max_points  # Maximum number of points to handle
        self.fc1 = nn.Linear(size * size, h1) # (batchsize, size*size) --> (batchsize, h1)
        self.fc2 = nn.Linear(h1, h2)  # (batchsize, h1) --> (batchsize, h2)
        self.out = nn.Linear(h2, 2 * max_points) # (batchsize, h2) --> (batchsize, 2*max_points)

    def forward(self, x):
        x = x.reshape(x.size(0), -1) # (batch_size, size, size) --> (batch_size, size * size)

        # Apply ReLU activation after the first fully connected layer
        x = F.relu(self.fc1(x)) 

        # Apply ReLU activation after the second fully connected layer  
        x = F.relu(self.fc2(x)) 

        # Output from the final fully connected layer
        x = self.out(x) 

        # Reshape the output to (batch_size, max_points, 2)
        # model_output[:, :, 0]: predicted amplitude
        # model_output[:, :, 1]: predicted phase
        return x.view(-1, self.max_points, 2) 


