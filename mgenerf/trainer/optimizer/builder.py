import copy
import inspect
from mgenerf.utils import Registry, build_from_cfg
import megengine.optimizer as mgeoptim
from megengine.autodiff import GradManager
import megengine.distributed as dist

OPTIMIZERS = Registry('optimizer')

def register_mge_optimizers():
    mge_optimizers = []
    for module_name in dir(mgeoptim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(mgeoptim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, mgeoptim.optimizer.Optimizer):
            OPTIMIZERS.register_module(_optim)
            mge_optimizers.append(module_name)
    return mge_optimizers

MGE_OPTIMIZERS = register_mge_optimizers() # 手动注册 不用装饰器


def build_optimizer(model, cfg):
    optimizer_cfg = copy.deepcopy(cfg)
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    from .default_constructor import DefaultOptimizerConstructor
    optim_constructor = DefaultOptimizerConstructor(optimizer_cfg=optimizer_cfg, paramwise_cfg=paramwise_cfg)
    optimizer = optim_constructor(model)
    return optimizer


def build_gradmanager(module):
    world_size = dist.get_world_size()
    gm = GradManager().attach(module.parameters(), callbacks=[dist.make_allreduce_cb("mean")] if world_size > 1 else None)
    return gm


def build_learning_rate_scheduler(optimizer, config, total_step):
    lr_scheduler = None
    learning_rate_type = config.type
    if learning_rate_type == "one_cycle":
        lr_scheduler = OneCycle(
            optimizer,
            total_step,
            config.lr_max,
            config.moms,
            config.div_factor,
            config.pct_start,
        )

    elif lr_scheduler is None:
        raise ValueError("Learning_rate %s not supported." % learning_rate_type)

    return lr_scheduler
