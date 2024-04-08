import numpy as np
import torch 
from .mnl_constants import *

class BaseModel(torch.nn.Module):
    def __init__(
        self, 
        in_channel, 
        in_feature,
        out_channel, 
        kernel=KERNEL_SIZE
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channel, out_channel, kernel)
        self.pool = torch.nn.MaxPool2d(kernel_size=POOLING_SIZE, stride=STRIDE)
        self.linear = torch.nn.Linear(in_feature, NUM_CLASSES, bias=False, dtype=torch.float)
        self.dropout = torch.nn.Dropout(DROPOUT)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        out = self.linear(x)
        return out
    
    def transform_single(self, x):
        """
        Passes the raw image through the forward pass UNTIL the linear layer.
        The output is the "x" in the paper.
        :param x: raw data
        :return: transformed 1D vector
        """
        with torch.no_grad():
            x = self.conv(x)
            x = torch.nn.functional.relu(x)
            x = self.pool(x)
            out = torch.flatten(x)
        return out
      
    def extract_features(dataloader, model, device):
        model.eval()  # Ensure model is in evaluation mode
        features = []
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(device)
                feature = model.transform_single(data)  # Assuming this returns a batch of features
                features.append(feature.detach().cpu())
        return torch.cat(features).numpy()

    def get_linear_weight(self):
        return self.linear.weight.detach()

    def set_linear_weight(self, weight):
        self.linear.weight = torch.nn.parameter.Parameter(weight)



