# interpolate in training poses
# save to gif
import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import argparse
import megengine.distributed as dist
from mgenerf.utils import get_root_logger, Config
from mgenerf.datasets import build_dataset
from mgenerf.models import build_nerf
from mgenerf.trainer import test_nerf
import time

def parse_args():
	parser = argparse.ArgumentParser(description="Test a nerf")
	parser.add_argument("config", help="test config file path")
	parser.add_argument("--work_dir", help="the dir to save logs and models")
	args = parser.parse_args()
	return args

# @dist.launcher
def main(timestamp):
	args = parse_args()
	cfg = Config.fromfile(args.config)

	if args.work_dir is not None:
		cfg.work_dir = args.work_dir

	logger = get_root_logger(cfg.log_level)

	cfg.work_dir = os.path.join(cfg.work_dir, timestamp)

	if dist.get_rank() == 0:
		backup_dir = os.path.join(cfg.work_dir, "configs")
		os.makedirs(backup_dir, exist_ok=True)
		try:
			os.system("cp %s %s/" % (args.config, backup_dir))
		except:
			pass
		logger.info(f"Backup config file to {cfg.work_dir}")

	dataset = build_dataset(cfg.data.test)

	assert len(cfg.workflow) == 1

	model = build_nerf(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

	logger.info(f"model structure: {model}")

	test_nerf(
		model,
		dataset,
		cfg,
		logger
	)

if __name__ == "__main__":
	timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
	main(timestamp)
