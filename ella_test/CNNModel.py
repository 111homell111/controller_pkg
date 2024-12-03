import torch
import torch.nn as nn
import torch.nn.functional as F

class DriveCNN(nn.Module):
    def __init__(self):
        super(DriveCNN, self).__init__()
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)  # Output: [32, 75, 100]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)  # Output: [64, 38, 50]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Output: [128, 19, 25]

        # Replace Adaptive Pooling with fixed pooling
        self.pool = nn.AvgPool2d(kernel_size=5, stride=5)  # Output: [128, 4, 5]

        # Update linear layer dimensions
        self.fc_shared = nn.Linear(128 * 3 * 5, 256)
        self.fc_linear = nn.Linear(256, 1)  # Outputs for linear velocity
        self.fc_angular = nn.Linear(256, 1)  # Outputs for angular velocity

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] to [B, C, H, W] 
        #print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        #print(x.shape)
        x = torch.flatten(x, 1)

        # Shared fully connected layer
        x = F.relu(self.fc_shared(x))

        # Separate outputs
        linear_out = self.fc_linear(x)
        angular_out = self.fc_angular(x)

        return linear_out, angular_out