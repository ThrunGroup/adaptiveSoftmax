import numpy as np
import torch 
from .mnl_constants import *

class BaseModel(torch.nn.Module):
    def __init__(
        self, 
        in_channel, 
        in_feature,
        out_channel=OUT_CHANNEL, 
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
        Runs forward pass of BaseModel EXCEPT the linear layer (equivalent to vector x in paper)
        :param x: raw data
        :return: transformed 1D vector
        """
        with torch.no_grad():
            x = self.conv(x)
            x = torch.nn.functional.relu(x)
            x = self.pool(x)
            out = torch.flatten(x)
        return out

    def get_prob(self, x):
      with torch.no_grad():
        x = self.forward(x)
        return torch.nn.functional.softmax(x)

    def get_linear_weight(self):
        return self.linear.weight.detach()

    def set_linear_weight(self, weight):
        self.linear.weight = torch.nn.parameter.Parameter(weight)


class TransformToLinear(object):
    """
    Defining transforms class to use torchvision transforms functionality
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
    def __call__(self, image):
        image = image.to(self.device)
        image = self.model.transform_single(image)

        return image


