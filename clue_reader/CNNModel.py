import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModelGray(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 3, 4096)  # Adjusted for the output of the convolutional layers
        self.fc2 = nn.Linear(4096, 36)  # Assuming 36 output classes (e.g., A-Z, 0-9)

        # Max Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        #x = x.permute(0, 3, 1, 2) #put channnel in the right place
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        #print(x.shape)

        # Flatten the tensor for fully connected layers
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch

        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x) # Output layer
        #no softmax cuz its in crossentropyloss already
        return x




#Relu = activation function. Adds non linearity
#Flatten for the fully connected layer which is similar to the MLP? output is 36 neurons corresponding to the different chars