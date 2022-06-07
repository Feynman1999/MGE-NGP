import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import argparse
import megengine.distributed as dist
from mgenerf.utils import get_root_logger, Config
from mgenerf.datasets import build_dataset
from mgenerf.models import build_nerf
from mgenerf.trainer import train_nerf

def parse_args():
    parser = argparse.ArgumentParser(description="Train a nerf")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument("--resume_from", help="the checkpoint file to resume from")
    args = parser.parse_args()
    return args

# @dist.launcher
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    cfg.gpus = dist.get_world_size()
    cfg.lr_config.lr_max = cfg.lr_config.lr_max * cfg.gpus
    logger = get_root_logger(cfg.log_level)

    if dist.get_rank() == 0:
        backup_dir = os.path.join(cfg.work_dir, "configs")
        os.makedirs(backup_dir, exist_ok=True)
        try:
            os.system("cp %s %s/" % (args.config, backup_dir))
        except:
            pass
        logger.info(f"Backup config file to {cfg.work_dir}/det3d")

    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))

    model = build_nerf(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    

    logger.info(f"model structure: {model}")

    train_nerf(
        model,
        datasets,
        cfg,
        logger
    )

if __name__ == "__main__":
    main()
