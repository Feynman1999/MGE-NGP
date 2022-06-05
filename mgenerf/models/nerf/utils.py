import megengine as mge
import megengine.functional as F
import numpy as np


def mge_search_sorted(cdf, u):
    """
        search by directly compare(bool)

        cdf: (N, sample - 1)
        u: (N, sample2)

        return : (N, sample2)   np.int32
    """
    cdf_N, N1 = cdf.shape
    N, N2 = u.shape
    assert cdf_N == N
    cdf = F.expand_dims(cdf, axis=1) # [N, 1, N1]
    cdf = F.broadcast_to(cdf, (N, N2, N1))
    u = F.expand_dims(u, axis=2) # [N, N2, 1]
    res = F.sum((u > cdf).astype(np.int32), axis=-1) # [N, N2]
    return res

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    """
        bins: N, N1-1  一阶段已经取的点的中点
        weights:  对应的权重  N, N1 - 2  (去掉首尾) 
        N_samples:  二阶段采样数 (N2)
        det: 
    """
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / F.sum(weights, -1, keepdims=True)
    cdf = F.cumsum(pdf, -1 + pdf.ndim) # (N, N1 - 2)
    cdf = F.concat([F.zeros_like(cdf[..., :1]), cdf], -1)  # (N, N1-1)

    # Take uniform samples
    if det:
        u = F.linspace(0.0, 1.0, N_samples)
        u = F.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = mge.random.uniform(size=list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = mge_search_sorted(cdf, u) # int
    below = F.maximum(F.zeros_like(inds), inds - 1)
    above = F.minimum((cdf.shape[-1] - 1) * F.ones_like(inds), inds)
    inds_g = F.stack([below, above], -1)  # (N, N2, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]] # [N, N_2, N_1-1]
    
    cdf_g = F.gather(F.broadcast_to(F.expand_dims(cdf, axis=1), matched_shape), 2, index = inds_g) # (N, N2, 2)
    bins_g = F.gather(F.broadcast_to(F.expand_dims(bins, axis=1), matched_shape), 2, index = inds_g) # (N, N2, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = F.where(denom < 1e-5, F.ones_like(denom), denom) # (N, N2)  防止除以0
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def cumprod(x: mge.Tensor, axis: int):
    dim = x.ndim
    axis = axis if axis > 0 else axis + dim
    num_loop = x.shape[axis]
    t_shape = [i + 1 if i < axis else i for i in range(dim)]
    t_shape[axis] = 0
    x = x.transpose(*t_shape)
    assert len(x) == num_loop
    cum_val = F.ones(x[0].shape)
    for i in range(num_loop):
        cum_val *= x[i]
        x[i] = cum_val
    return x.transpose(*t_shape)