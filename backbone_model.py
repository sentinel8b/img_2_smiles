import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B3_Weights
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.utils as torch_utils
import torch
from efficientnet_v2 import EfficientNetV2


class ImgEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = models.efficientnet_b3(weights = 'DEFAULT').features
        self.encoder = models.efficientnet_b3().features
        # self.encoder = models.efficientnet_b3().features
    
    def forward(self, X):
        out = self.encoder(X)
        # print(out.shape)
        out = out.view(out.shape[0], out.shape[1], -1)
        # print(out.shape)
        out = torch.permute(out, (0,2,1))
        return out


# encoder = ImgEncoder(232)
# sample_tensor = torch.ones(2,3,299,299)

# result = encoder.forward(sample_tensor)
# print(result.shape)