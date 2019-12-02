import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F

class SampleSoftmaxLayer(nn.Module):
    def __init__(self, squeeze_channels = False, smooth=0, **kwargs):
        self.squeeze_channels = squeeze_channels
        self.smooth = smooth
        super(SampleSoftmaxLayer, self).__init__(**kwargs)

    # TODO: Read more about the build method and what it has to do

    def forward(self, x):
        # Apply softmax to the whole x (per sample)
        input_shape = x.shape
        x = nn.Softmax(x.view(input_shape[0], -1))

        # smooth the distribution ? what distribution
        if 0 < self.smooth < 1:
            x = x*(1-self.smooth)
            x = x + self.smooth / torch.FloatTensor(x.shape[1])

        # Finally reshape to the original shape
        x = x.reshape(input_shape)

        if self.squeeze_channels:
            # assuming the the channel dimension is at the end
            channels = -1
            x = torch.squeeze(x, channels)
        
        return x

    # helper function
    def compute_output_shape(self, input_shape):
        if not self.squeeze_channels:
            return input_shape
        shape = list(input_shape)
        # remove the channels 
        # TODO: Please note that the
        channels = -1
        shape.pop(channels)
        return tuple(shape)