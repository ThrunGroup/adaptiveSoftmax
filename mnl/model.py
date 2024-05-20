import numpy as np
import torch
import torch.nn as nn
from .mnl_constants import *

class BaseModelMNIST(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BaseModelMNIST, self).__init__()

        # first block
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel, 
                out_channels=out_channel, 
                kernel_size=CONST_KERNEL_SIZE, 
                padding=CONST_PADDING,
                stride=CONST_STRIDE,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
            nn.MaxPool2d(kernel_size=POOLING)
        )

        # second block
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=out_channel, 
                out_channels=out_channel * 2, 
                kernel_size=CONST_KERNEL_SIZE, 
                padding=CONST_PADDING,
                stride=CONST_STRIDE
            ),     
            nn.ReLU(),        
            nn.BatchNorm2d(out_channel * 2),              
            nn.MaxPool2d(2),                
            nn.Dropout(p=0.2)
        )
        self.linear = None  

    def _transform_single(self, x):
        """
        Passes the raw image through the forward pass UNTIL the linear layer.
        The output is the "x" in the paper.
        :param x: raw data
        :return: transformed 1D vector
        """
        x = self.conv1(x)
        x = self.conv2(x)
        out = torch.flatten(x, start_dim=1)
        return out
        
    def forward(self, x):
        x = self._transform_single(x)
        if self.linear is None:
            self.linear = torch.nn.Linear(
                x.shape[1], 
                NUM_CLASSES, 
                bias=False, 
                dtype=torch.float
            ).to(x.device)
            print(self.linear)
        
        out = self.linear(x)
        return out
      
    def extract_features(self, dataloader, device):
        self.eval() 
        features = []
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(device)
                feature = self._transform_single(data)  
                features.append(feature.detach().cpu())
        return torch.cat(features).numpy()

    def get_linear_weight(self):
        return self.linear.weight.detach()


class CustomVGG(nn.Module):
    def __init__(self, vgg_model, blocks_not_freezing=BLOCKS_NOT_FREEZING):
        super(CustomVGG, self).__init__()
        self.features = vgg_model.features
        self.avgpool = vgg_model.avgpool
        self.classifier = nn.Linear(VGG19_IN_FEATURES, NUM_CLASSES)
        self._freeze_weights(blocks_not_freezing * 3)  # single block is Conv2d -> ReLU -> MaxPool2d

    def _freeze_weights(self, layers_not_freezing):
        for i, layer in enumerate(self.features):
            if i < len(self.features) - layers_not_freezing:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def _transform_single(self, x):
        """
        Passes the raw image through the forward pass UNTIL the linear layer.
        The output is the "x" in the paper.
        :param x: raw data
        :return: transformed 1D vector
        """
        x = self.features(x)
        x = self.avgpool(x)
        out = torch.flatten(x, 1)
        return out
    
    def forward(self, x):
        x = self._transform_single(x)
        x = self.classifier(x)
        return x
    
    def extract_features(self, dataloader, device):
        self.eval() 
        features = []
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(device)
                feature = self._transform_single(data)  
                features.append(feature.detach().cpu())
        return torch.cat(features).numpy()

    def get_linear_weight(self):
        return self.classifier.weight.data
    

