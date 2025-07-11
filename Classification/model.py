import torch
import torch.nn as nn
import torch.nn.functional as F

class GenreCNN(nn.Module):
    """
    CNN for genre classification with dropout regularization after fc1.
    """
    def __init__(self, num_classes=10):
        super(GenreCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def set_num_classes(self, num_classes):
        self.fc2 = nn.Linear(self.fc2.in_features, num_classes)
