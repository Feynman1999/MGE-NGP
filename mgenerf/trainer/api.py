from mgenerf.datasets import build_dataloader
from .trainer import Trainer
from .optimizer import build_optimizer, build_gradmanager, build_onecyclelr
import megengine.distributed as dist


def train_nerf(model, dataset, cfg, logger):
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [build_dataloader(dataset[0], cfg.data.samples_per_gpu, cfg.data.workers_per_gpu, shuffle=True)]
    if len(dataset) > 1:
        data_loaders.append(build_dataloader(dataset[1], cfg.data.samples_per_gpu, cfg.data.workers_per_gpu, shuffle=False))

    total_steps = cfg.total_epochs * len(data_loaders[0])
    
    logger.info("total_epochs, steps per epoch, total_steps: {}, {}, {}".format(cfg.total_epochs, len(data_loaders[0]), total_steps))

    # dist.bcast_list_(model.tensors())

    optimizer = build_optimizer(model, cfg.optimizer)
    grad_manager = build_gradmanager(model)
    lr_scheduler = build_onecyclelr(optimizer, cfg.lr_config, total_steps)
    
    trainer = Trainer(model, optimizer, grad_manager, cfg.work_dir, logger, lr_scheduler)

    trainer.register_training_hooks(cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        trainer.resume(cfg.resume_from)
    elif cfg.load_from:
        trainer.load_checkpoint(cfg.load_from)

    trainer.run(data_loaders, cfg.workflow, cfg.total_epochs)


def test_nerf(model, dataset, cfg, logger):
    data_loaders = [build_dataloader(dataset, 1, cfg.data.workers_per_gpu, shuffle=False)]

    trainer = Trainer(model, None, None, cfg.work_dir, logger)

    trainer.load_checkpoint(cfg.load_from)
    
    trainer.run(data_loaders, cfg.workflow, 1)
