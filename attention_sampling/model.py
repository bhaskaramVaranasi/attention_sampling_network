import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F

from .layers.SampleSoftmaxLayer import SampleSoftmaxLayer
from .layers.L2NormalizeLayer import L2NormalizeLayer

def multinomial_entropy(*args, **kwargs):
    pass 

class FromTensors:
    # TODO: implement method
    pass

class SamplePatches:
    def __init__(self,n_patches, patch_size, receptive_field =0, replace= False, use_logits = False, **kwargs):
        self._n_patches = n_patches
        self._patch_size = patch_size
        self._receptive_field = receptive_field
        self._replace = replace
        self._use_logits = use_logits

        super(SamplePatches, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        shape_low, shape_high, shape_att = input_shape

        # Figure out the shape of the patches
        # we always assume that the channels are at the end
        patch_shape = (*self._patch_size, shape_high[-1])
        patches_shape = (shape_high[0], self._n_patches, *patch_shape)

        #sampled attention
        att_shape = (shape_high[0], self._n_patches)

        return [patches_shape, att_shape]
    
    def forward(self, x):
        x_low, x_high, attnetion = x
        
        sample_space = attnetion.shape[1:]
        samples, sampled_attention = sample(
            self._n_patches,
            attnetion,
            sample_space,
            replace=self._replace,
            use_logits= self._use_logits
        )

        offsets = torch.zeros_like(samples, dtype=torch.float32)
        if self._receptive_field > 0:
            offsets = offsets + self._receptive_field / 2
        
        patches, _ = FromTensors([x_low, x_high], None).patches(
            samples,
            offsets,
            sample_space,
            x_low.shape[1: -1] - self._receptive_field,
            self._patch_size,
            0,
            1
        )

        return [patches, sampled_attention]

def attention_sampling(attention, feature, patch_size=None, n_patches = 10, replace = False, attention_regularizer=None, receptive_field=0):
    """Use attention sampling to process a high resolution image in patches.
    This function is meant to be a convenient way to use the layers defined in
    this module with Keras models or callables.
    Arguments
    ---------
        attention: A Keras layer or callable that takes a low resolution tensor
                   and returns and attention tensor
        feature: A Keras layer or callable that takes patches and returns
                 features
        patch_size: Tuple or tensor defining the size of the patches to be
                    extracted. If not given we try to extract it from the input
                    shape of the feature layer.
        n_patches: int that defines how many patches to extract from each
                   sample
        replace: bool, whether we should sample with replacement or without
        attention_regularizer: A regularizer callable for the attention
                               distribution
        receptive_field: int, how large is the receptive field of the attention
                         network. It is used to map the attention to high
                         resolution patches.
    Returns
    -------
        In the spirit of Keras we return a function that expects two tensors
        and returns three, namely the `expected features`, `attention` and
        `patches`
        ([x_low, x_high]) -> [expected_features, attention, patches]
    """
    if receptive_field is None:
        raise NotImplementedError(("Receptive field is not implemented yet"))
    
    if patch_size is None:
        # we need the patch size to be present
        # if needed we will add code to cover this functionality
        raise NotImplementedError("Patch size cannot be None")

    def apply_attention_sampling(x):
        assert isinstance(x, list) and len(x) == 2
        x_low, x_high = x

        # first compute the attention map
        attention_map = attention(x_low)
        if attention_regularizer is not None:
            attention_map = ActivityRegularizer(attention_regularizer)(attention_map)

        # Then we sample patches based on the attention
        patches, sampled_attention = SamplePatches(
            n_patches,
            patch_size,
            receptive_field,
            replace
        )([x_low, x_high, attention_map])

        # we compute the features of the sampled patches
        channels = patches.shape[-1]
        patches_flat = TotalReshape((-1, *patch_size, channels))(patches)
        patch_features = feature(patches_flat)
        dims = patch_features.shape[-1]
        patch_features = TotalReshape((-1, n_patches, dims))(patch_features)

        sample_features = Expectation(replace)([
            patch_features,
            sampled_attention
        ])
        return [sample_features, attention_map, patches]
    
    return apply_attention_sampling


def get_model(outputs, width, height, scale, n_patches, patch_size, reg):
    '''
    Creates and returns the Network 
    '''

    # define the shapes for the high and low image sizes
    shape_high = (height, width, 1)
    scaled_height, scaled_witdh = [x * scale for x in [height, width]]
    shape_low = (scaled_height, scaled_witdh, 1)

    # TODO: Make sure to check the size compatabilites by testing
    #       using colab machines
    attention = nn.Sequential(
        nn.Conv2d(shape_low, 8, kernel_size=3, padding='same'),
        nn.Tanh(),
        nn.Conv2d(8, 8, kernel_size=3, padding='same'),
        nn.Tanh(),
        nn.Conv2d(8, 1, kernel_size=3, padding='same'),
        SampleSoftmaxLayer(squeeze_channels=True, smooth=1e-5)
    )

    feature = nn.Sequential(
        nn.Conv2d(shape_high, 32, kernel_size)
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), # https://stackoverflow.com/questions/52622518/how-to-convert-pytorch-adaptive-avg-pool2d-method-to-keras-or-tensorflow
        L2NormalizeLayer()
    )

    #TODO: yet to be replaced
    # this will change drastically
    # hence not changing it now
    features, attention, patches = attention_sampling(
        attention,
        feature,
        patch_size,
        n_patches,
        replace=False,
        attention_regularizer=multinomial_entropy(reg)
    )([x_low, x_high])
    y = nn.Linear(outputs, activation="softmax")(features)
    
    # TODO: decide on what to be returned
