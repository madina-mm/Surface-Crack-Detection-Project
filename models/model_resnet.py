import torch
import torch.nn as nn
import torchvision.models as models


class ModelRes(nn.Module):
    
    def __init__(self, n_classes):
        super(ModelRes, self).__init__()
        
        # Load the pretrained ResNet18 model
        resnet = models.resnet18(pretrained=True)
        
        # Replace the fully connected layer with a new one
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, n_classes)
        
        # Set the modified ResNet18 as the feature extractor
        self.feature_extractor = resnet
        
        # Softmax layer for classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.softmax(x)

        return x
