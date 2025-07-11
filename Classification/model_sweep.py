import torch
import torch.nn as nn
import torch.nn.functional as F

class GenreCNN(nn.Module):
    """
    CNN for genre classification with configurable kernel sizes for sweeping.
    """
    def __init__(self, num_classes=10, kernel_config=None):
        super(GenreCNN, self).__init__()
        
        # Default kernel configuration
        if kernel_config is None:
            kernel_config = {
                'conv1_kernel': (3, 3),
                'conv2_kernel': (3, 3),
                'conv1_padding': (1, 1),
                'conv2_padding': (1, 1)
            }
        
        # Extract kernel sizes and padding
        conv1_kernel = kernel_config['conv1_kernel']
        conv2_kernel = kernel_config['conv2_kernel']
        conv1_padding = kernel_config['conv1_padding']
        conv2_padding = kernel_config['conv2_padding']
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=conv1_kernel, padding=conv1_padding)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=conv2_kernel, padding=conv2_padding)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the size after convolutions and pooling
        # This is a simplified calculation - you might need to adjust based on your input size
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Store config for logging
        self.kernel_config = kernel_config

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

# Predefined kernel configurations to test
KERNEL_CONFIGS = {
    'small': {
        'conv1_kernel': (3, 3),
        'conv2_kernel': (3, 3),
        'conv1_padding': (1, 1),
        'conv2_padding': (1, 1)
    },
    'medium': {
        'conv1_kernel': (5, 5),
        'conv2_kernel': (3, 3),
        'conv1_padding': (2, 2),
        'conv2_padding': (1, 1)
    },
    'large': {
        'conv1_kernel': (7, 7),
        'conv2_kernel': (5, 5),
        'conv1_padding': (3, 3),
        'conv2_padding': (2, 2)
    },
    'asymmetric_time': {
        'conv1_kernel': (3, 7),
        'conv2_kernel': (3, 5),
        'conv1_padding': (1, 3),
        'conv2_padding': (1, 2)
    },
    'asymmetric_freq': {
        'conv1_kernel': (7, 3),
        'conv2_kernel': (5, 3),
        'conv1_padding': (3, 1),
        'conv2_padding': (2, 1)
    },
    'mixed': {
        'conv1_kernel': (5, 7),
        'conv2_kernel': (3, 5),
        'conv1_padding': (2, 3),
        'conv2_padding': (1, 2)
    }
} 