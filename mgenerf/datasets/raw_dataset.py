import cv2
from mgenerf.datasets.registry import DATASETS
from mgenerf.datasets.base_dataset import BaseDataset
import os
import numpy as np
import glob
from progress.bar import Bar
import random


@DATASETS.register_module
class RawSRDataset(BaseDataset):
	def __init__(self, pipeline, root_path, mode):
		super().__init__(pipeline, mode)
		self.root_path = root_path
		self.load_annotations()

	def load_annotations(self):
		hr_infos = [] # list of dict
		lr_infos = []

		folder_paths = os.listdir(self.root_path)

		for folder in sorted(folder_paths):
			if self.mode == 'train':
				# get lr_path and hr_path
				# hr (f0 f1) *450   lr  (f5 f6) * 450
				
				hr_infos.append({
					'hr_path' : os.path.join(self.root_path, folder, '00001.ARW')
				})
				hr_infos.append({
					'hr_path' : os.path.join(self.root_path, folder, '00002.ARW')
				})
				lr_infos.append({
					'lr_path' : os.path.join(self.root_path, folder, '00006.ARW')
				})
				lr_infos.append({
					'lr_path' : os.path.join(self.root_path, folder, '00007.ARW')
				})

			elif self.mode == 'eval':
				pass
			else: # test
				pass

		self.lr_infos = lr_infos
		self.hr_infos = hr_infos


	def shuffle_all_rays(self):
		print("Shuffle all images")
		# need to set seed for multi gpu
		random.shuffle(self.lr_infos)


	def __getitem__(self, idx):
		# given idx 
		# get some rays for train (map idx to stationary idx)
		# [idx * rays_per_sample ~  (idx+1) * rays_per_sample]
		
		res_dict = {
			'lr_path' : self.lr_infos[idx]['lr_path'],
			'hr_path' : self.hr_infos[idx]['hr_path']   
		}
		
		return self.pipeline(res_dict) # to img and crop

	def __len__(self):
		"""Length of the dataset.
		Returns:
			int: Length of the dataset.
		"""
		return len(self.lr_infos)

	def evaluate(self, results):
		assert self.mode == "eval"
		pass
