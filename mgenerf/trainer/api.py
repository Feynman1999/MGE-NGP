from mgenerf.datasets import build_dataloader
from .trainer import Trainer
from .optimizer import build_optimizer, build_gradmanager, build_onecyclelr


def train_nerf(model, dataset, cfg, logger):
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [build_dataloader(dataset[0], cfg.data.samples_per_gpu, cfg.data.workers_per_gpu, shuffle=True)]
    if len(dataset) > 1:
        data_loaders.append(build_dataloader(dataset[1], cfg.data.samples_per_gpu, cfg.data.workers_per_gpu, shuffle=False))

    total_steps = cfg.total_epochs * len(data_loaders[0])
    
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
