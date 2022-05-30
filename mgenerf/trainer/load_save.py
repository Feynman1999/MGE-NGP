import os.path as osp
import megengine
from collections import OrderedDict

from numpy import mask_indices
from mgenerf.utils import get_dist_info, master_only
from mgenerf.utils.path import mkdir_or_exist


@master_only
def save_checkpoint(model, filename, optimizer=None, meta=None):
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError("meta must be a dict or None, but got {}".format(type(meta)))

    mkdir_or_exist(osp.dirname(filename))
    if hasattr(model, "module"):
        model = model.module

    checkpoint = {"meta": meta, "state_dict": model.state_dict()}
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()

    megengine.save(checkpoint, filename)


@master_only
def load_checkpoint(model, filename, strict=False):
    if not osp.isfile(filename):
        raise IOError("{} is not a checkpoint file".format(filename))

    checkpoint = megengine.load(filename)

    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        raise RuntimeError("No state_dict or model found in checkpoint file {}".format(filename))

    # load state_dict
    model.load_state_dict(state_dict, strict = strict)
    return checkpoint
