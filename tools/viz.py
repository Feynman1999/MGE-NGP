import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import argparse
from mgenerf.utils import get_root_logger, Config
from mgenerf.datasets import build_dataset
from mgenerf.datasets import build_dataloader
import numpy as np

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
	parser = argparse.ArgumentParser(description="viz camera poses")
	parser.add_argument("config", help="train config file path")
	args = parser.parse_args()
	return args


def get_rays_np(H, W, K, c2w):
	"""
		K: intrinstic of camera [fu fv cx cy]
		c2w: camera to world transformation
	"""
	i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
	dirs = np.stack([(i-K[2])/K[0], (j-K[3])/K[1], np.ones_like(i), np.ones_like(i)], -1)  #  [h,w,4]
	dirs = np.reshape(dirs, (H*W, 4, 1))

	# Rotate ray directions from camera frame to the world frame

	c2w = np.concatenate([c2w, [[0,0,0,1]]], axis=0) # [4,4]
	c2w = c2w.reshape(1, 4, 4)

	rays_d = np.matmul(c2w, dirs)[:, :, 0]
	rays_d = rays_d.reshape(H, W, 4)[:, :, :3]
	# Translate camera frame's origin to the world frame. It is the origin of all rays.
	rays_o = np.broadcast_to(c2w[0, :3,-1], np.shape(rays_d))
	return rays_o, rays_d # [h,w,3]  [h,w,3]


def add_line(start, end, ax, num = 2):
	points = start + np.linspace(0, 1, num).reshape(num, 1) * (end - start)

	ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'gray')

def main():
	args = parse_args()
	cfg = Config.fromfile(args.config)
	logger = get_root_logger(cfg.log_level)
	dataset = build_dataset(cfg.data.test)
	data_loader = build_dataloader(dataset, cfg.data.samples_per_gpu, cfg.data.workers_per_gpu, shuffle=False)


	fig = plt.figure()
	ax = plt.axes(projection="3d")

	for item in data_loader:
		_, h, w ,_ = item['img'].shape
		intrinsics = item['intrinsics'][0] # [4, ]
		pose = item['pose'][0] # [3, 4]

		rays_o, rays_d = get_rays_np(h, w, intrinsics, c2w = pose)

		add_line(rays_o[0, 0:1, :], rays_d[0, 0:1, :], ax = ax)
		add_line(rays_o[0, 0:1, :], rays_d[0, w-1:w, :], ax = ax)
		add_line(rays_o[0, 0:1, :], rays_d[h-1, 0:1, :], ax = ax)
		add_line(rays_o[0, 0:1, :], rays_d[h-1, w-1:w, :], ax = ax)
		add_line(rays_d[0, 0:1, :], rays_d[0, w-1:w, :], ax = ax)
		add_line(rays_d[0, 0:1, :], rays_d[h-1, 0:1, :], ax = ax)
		add_line(rays_d[h-1, w-1:w, :], rays_d[0, w-1:w, :], ax = ax)
		add_line(rays_d[h-1, w-1:w, :], rays_d[h-1, 0:1, :], ax = ax)


	plt.show()

	
if __name__ == "__main__":
	main()
