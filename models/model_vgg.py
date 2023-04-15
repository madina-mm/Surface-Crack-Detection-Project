import torch
import torch.nn as nn
import torchvision.models as models


class ModelVGG(nn.Module):
    def __init__(self, num_classes):
        super(ModelVGG, self).__init__()

        # Load the pretrained VGG16 model
        vgg16 = models.vgg16(pretrained=True)

        # Replace the fully connected layer with a new one
        vgg16.classifier[6] = nn.Linear(4096, num_classes)

        # Set the modified VGG16 as the feature extractor
        self.features = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        # Softmax layer for classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.softmax(x)

        return x


# class ModelVGG(nn.Module):
#     def __init__(self, num_classes):
#         super(ModelVGG, self).__init__()
        
#         # Load the pretrained VGG16 model
#         vgg16 = models.vgg16(pretrained=True)
        
#         # Replace the fully connected layer with a new one
#         vgg16.classifier[6] = nn.Linear(4096, num_classes)
        
#         # Set the modified VGG16 as the feature extractor
#         self.features = vgg16.features
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.classifier = vgg16.classifier[:-1]
#         self.fc = nn.Linear(4096, num_classes)
        
#         # Softmax layer for classification
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         x = self.fc(x)
#         x = self.softmax(x)

#         return x
    


# class ModelVGG(nn.Module):
    
#     def __init__(self, n_classes):
#         super(ModelVGG, self).__init__()
        
#         # Load the pretrained VGG16 model
#         vgg16 = models.vgg16(pretrained=True)
        
#         # Replace the fully connected layer with a new one
#         num_ftrs = vgg16.classifier[6].in_features
#         vgg16.classifier[6] = nn.Linear(num_ftrs, n_classes)
        
#         # Set the modified VGG16 as the feature extractor
#         self.feature_extractor = vgg16.features
        
#         # Softmax layer for classification
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.feature_extractor(x)
#         x = x.view(x.size(0), -1)
#         x = self.feature_extractor(x)
#         x = self.softmax(self.fc(x))

#         return x
    
