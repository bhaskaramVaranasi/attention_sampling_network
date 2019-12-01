import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F

class L2NormalizeLayer(nn.Module):
    def __init__(self, axis = -1, **kwargs):
        self.axis = axis
        super(L2Normalize, self).__init__(**kwargs)

    def forward(self, x):
        # https://discuss.pytorch.org/t/how-to-normalize-embedding-vectors/1209/7
        norm = x.norm(p = 2, dim = self.axis, keepdim= True)
        return x.div(norm)