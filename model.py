import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels = 6,
                kernel_size = 5,
                stride = 1,
                padding = 2,
            ),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
        )
        
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x