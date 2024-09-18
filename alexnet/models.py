import torch
import torch.nn as nn 

class AlexNet(nn.Module):
    def __init__(self, no_of_classes):
        super(AlexNet, self).__init__()
        
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=1),  # N x 96 x 55 x 55,
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2), # N x 96 x 27 x 27,
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5),  # N x 256 x 23 x 23
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),  # N x 256 x 11 x 11
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=348, kernel_size=3),  # N x 348 x 9 x 9
            nn.ReLU(),
            nn.Conv2d(in_channels=348, out_channels=348, kernel_size=3), # N x 348 x 7 x 7
            nn.ReLU(),
            nn.Conv2d(in_channels=348, out_channels=256, kernel_size=3),  # N x 348 x 5 x 5
            nn.MaxPool2d(kernel_size=3, stride=2), # N x 256 x 2 x 2
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(), # N x 1024
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(256 * 2 * 2), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Softmax()
        )
        self.init_parameter()
        
    def init_parameter(self):
        for layer in self.convs:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.1),
                nn.init.constant_(layer.bias, 0)
            
        nn.init.constant_(self.convs[4].bias, 1)
        nn.init.constant_(self.convs[10].bias, 1)
        nn.init.constant_(self.convs[12].bias, 1)
        nn.init.constant_(self.classifier[2].bias, 1)
        nn.init.constant_(self.classifier[5].bias, 1)
        nn.init.constant_(self.classifier[7].bias, 1)
        
    
    def forward(self, x):
        x = self.convs(x)
        x = self.classifier(x)
        return x
        