import numpy as np
import torch
import torch.nn as nn
from .mnl_constants import *

class BaseModel(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BaseModel, self).__init__()

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
            nn.MaxPool2d(2),                
        )
        self.linear = None  
        
    def forward(self, x):
        x = self.transform_single(x)
        if self.linear is None:
            self.linear = torch.nn.Linear(
                x.shape[1], 
                NUM_CLASSES, 
                bias=False, 
                dtype=torch.float
            ).to(x.device)
        
        out = self.linear(x)
        return out
    
    def transform_single(self, x):
        """
        Passes the raw image through the forward pass UNTIL the linear layer.
        The output is the "x" in the paper.
        :param x: raw data
        :return: transformed 1D vector
        """
        x = self.conv1(x)
        # x = self.conv2(x)
        out = torch.flatten(x, start_dim=1)
        return out
      
    def extract_features(self, dataloader, device):
        self.eval() 
        features = []
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(device)
                feature = self.transform_single(data)  
                features.append(feature.detach().cpu())
        return torch.cat(features).numpy()

    def get_linear_weight(self):
        return self.linear.weight.detach()

    def set_linear_weight(self, weight):
        self.linear.weight = torch.nn.parameter.Parameter(weight)



