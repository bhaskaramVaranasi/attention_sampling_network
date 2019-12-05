


def _sample_with_replacement(logits):
    """Sample with replacement using the tensorflow op."""
    return Categorical(logits=logits)

def _sample_without_replacement(logits, n_samples):
    """Sample without replacement using the Gumbel-max trick.

    See lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
    """
    z = -torch.log(-torch.log(Uniform(logits.shape)))
    return torch.topk(logits+z, k=n_samples, dim=0)[1]

def gather_nd(params, indices, name=None):
    '''
    the input indices must be a 2d tensor in the form of [[a,b,..,c],...], 
    which represents the location of the elements.
    '''
    
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1
    
    for i in range(ndim)[::-1]:
        idx += indices[i] * m 
        m *= params.size(i)
    
    return torch.take(params, idx)


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def sample(n_samples, attention, sample_space, replace=False,
           use_logits=False):
    """Sample from the passed in attention distribution.

    Arguments
    ---------
        n_samples: int, the number of samples per datapoint
        attention: tensor, the attention distribution per datapoint (could be
                   logits or normalized)
        sample_space: This should always equal K.shape(attention)[1:]
        replace: bool, sample with replacement if set to True (defaults to
                 False)
        use_logits: bool, assume the input is logits if set to True (defaults
                    to False)
    """
    # Make sure we have logits and choose replacement or not
    logits = attention if use_logits else torch.log(attention)
    sampling_function = (
        _sample_with_replacement if replace
        else _sample_without_replacement
    )

    # Flatten the attention distribution and sample from it
    logits = logits.view((-1, torch.prod(sample_space)))
    samples = sampling_function(logits, n_samples)

    
    # Unravel the indices into sample_space
    batch_size = attention.shape[0]
    n_dims = sample_space.shape[0]
    samples =  unravel_index(samples.view((-1,)), sample_space)
    samples = samples.transpose().view((batch_size, n_samples, n_dims))

    # Concatenate with the indices into the batch dimension in order to gather
    # the attention values
    batch_indices = (
        torch.arange(0,batch_size).view((-1, 1, 1)) *
        torch.ones((1, n_samples, 1)).int()
    )
    indices = torch.cat((batch_indices, samplesa), 0)

    # Gather the attention
    sampled_attention = gather_nd(attention, indices)

    return samples, sampled_attention