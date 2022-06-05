import cv2
from mgenerf.datasets.registry import DATASETS
from mgenerf.datasets.base_dataset import BaseDataset
import os
import numpy as np
import glob
from .camera import CameraPoseTransform
from progress.bar import Bar


def get_rays_np(H, W, K, c2w):
    """
        K: intrinstic of camera [fu fv cx cy]
        c2w: camera to world transformation
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[2])/K[0], -(j-K[3])/K[1], -np.ones_like(i)], -1)  #  [h,w,3]
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d # [h,w,3]  [h,w,3]


@DATASETS.register_module
class KittiDataset(BaseDataset):
    def __init__(self, pipeline, root_path, rays_per_sample = 1024, consider_imgs = 50, mode='train'):
        super().__init__(pipeline, mode)
        self.root_path = root_path
        self.rays_per_sample = rays_per_sample
        self.consider_imgs = consider_imgs
        self.load_annotations()

    def load_annotations(self):
        calib_path = os.path.join(self.root_path, "calib.txt")
        pose_path = os.path.join(self.root_path, "poses.txt")
        image_left_path = sorted(glob.glob(self.root_path +  "/image_left/*.png"))
        n_frames = len(image_left_path)
        if n_frames > self.consider_imgs:
            image_left_path = image_left_path[: self.consider_imgs]
            n_frames = self.consider_imgs
            print('only consider first {} imgs'.format(n_frames))

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

                if len(poses) == n_frames:
                    break

        poses = np.stack(poses) # [N, 3, 4]
        poses_left = poses.copy()

        # 参考 https://github.com/utiasSTARS/pykitti/blob/master/pykitti/odometry.py
        baseline_left = np.append(-calib[2, 4:] / calib[2, 0], 1.0)  # (4, )
        poses_left[:, :, 3] = poses @ baseline_left # (N, 3, 4)

        # poses_left = [
        #         CameraPoseTransform.get_pose_from_matrix(pose)
        #         for pose in poses_left
        #     ]
        
        intrinsics_left = calib[2, :4] # (4, )

        with Bar('getting all images', max=n_frames) as bar:
            images = []
            for path in image_left_path:
                images.append(cv2.imread(path))
                bar.next()
            images = np.stack(images, 0) / 255. # [N, H, W, 3]

        _, H,W,_ = images.shape

        with Bar('getting all rays', max=n_frames) as bar:
            rays = []
            for p in poses_left:
                rays.append(get_rays_np(H, W, K = intrinsics_left, c2w = p) )
                bar.next()
            rays = np.stack(rays, 0) # [N, ro+rd, H, W, 3]

            rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
            rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
            rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [N*H*W, ro+rd+rgb, 3]
            rays_rgb = rays_rgb.astype(np.float32)

        self.rays_rgb = rays_rgb
        self.total_rays = rays_rgb.shape[0]
        self.intrinsics = intrinsics_left

    def evaluate(self, results):
        assert self.mode == "eval"
        pass

    def shuffle_all_rays(self):
        print("Shuffle all rays!")
        # need to set seed for multi gpu
        self.rays_rgb = np.random.shuffle(self.rays_rgb)

    def __getitem__(self, idx):
        # given idx 
        # get some rays for train (map idx to stationary idx)
        # [idx * rays_per_sample ~  (idx+1) * rays_per_sample]
        start = idx * self.rays_per_sample
        sample   = self.rays_rgb[start : start + self.rays_per_sample] # [x, 3, 3]
        sample = np.transpose(sample, (1, 0, 2)) # [3, x, 3]

        res_dict = {
            'rays' : sample[:2], # [2, x, 3]  
            'target' : sample[2],  # [x, 3] 
            'intrinsics'  : self.intrinsics  # [4, ]        
        }
        
        return res_dict

    def __len__(self):
        """Length of the dataset.
        Returns:
            int: Length of the dataset.
        """
        return self.total_rays // self.rays_per_sample 
