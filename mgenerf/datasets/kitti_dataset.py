from ast import Not
from mgenerf.datasets.registry import DATASETS
from mgenerf.datasets.base_dataset import BaseDataset
import os
import numpy as np
import glob
from .camera import CameraPoseTransform

@DATASETS.register_module
class KittiDataset(BaseDataset):
    def __init__(self, pipeline, root_path, mode='train'):
        super().__init__(pipeline, mode)
        self.root_path = root_path
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """
            load all data to memery

            include:
            * images
            * calib
            * poses
        """
        calib_path = os.path.join(self.root_path, "calib.txt")
        pose_path = os.path.join(self.root_path, "poses.txt")
        image_left_path = sorted(glob.glob(self.root_path +  "/image_left/*.png"))
        n_frames = len(image_left_path)

        with open(calib_path, "r") as fr:
            calib = np.loadtxt(fr, usecols=(1, 6, 3, 7, 4, 8, 12)) # fu fv cx cy x x x (0 0 0 for p0)  # [5,7]
            
        poses = []
        with open(pose_path, "r") as f:
            for line in f.readlines():
                ans = []
                for item in line.split(" "):
                    ans.append(float(item))
                if len(ans) != 12:
                    continue
                poses.append(np.array(ans).reshape(3,4))
        poses = np.stack(poses) # [N, 3, 4]
        poses_left = poses.copy()

        # 参考 https://github.com/utiasSTARS/pykitti/blob/master/pykitti/odometry.py
        baseline_left = np.append(-calib[2, 4:] / calib[2, 0], 1.0)  # (4, )
        poses_left[:, :, 3] = poses @ baseline_left # (N, 3)

        poses_left = [
                CameraPoseTransform.get_pose_from_matrix(pose)
                for pose in poses_left
            ]
        
        intrinsics_left = calib[2, :4] # (4, )

        data_infos = [] # 每一帧的信息
        for idx in range(n_frames):
            if self.mode in ['train', 'eval']:
                res_dict = {
                    "idx": idx,
                    "n_frames": n_frames,
                    "intrinsic_left": intrinsics_left, # (4, )
                    "image_left_path": image_left_path[idx],
                    "pose_left": poses_left[idx], # (7,)
                }
                data_infos.append(res_dict)
            else: # test
                raise NotImplementedError("")
        
        return data_infos
        
    def evaluate(self, results):
        pass