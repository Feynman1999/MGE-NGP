"""
	test extrinsic optimize
	given two points and initial extrinsic
"""
import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import megengine
import megengine.functional as F
from mgenerf.models.extrinsic import ExtrinsicOptimizer
from megengine.autodiff import GradManager
import megengine.optimizer as optim
import numpy as np

num_views = 1
model = ExtrinsicOptimizer(num_views)

gm = GradManager().attach(model.parameters())
optimizer = optim.Adam(model.parameters(), lr=0.01)  # lr may vary with different model

init_extrinsic = F.eye(4)[None] # [1 ,4, 4]
idx = megengine.tensor([0], dtype=np.int32)

source_points  = megengine.tensor([[0, 0, 0, 1], [1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1]]) # [4,4]
batch = len(source_points)
gt_str = "9.999978e-01 5.272628e-04 -2.066935e-03 -4.690294e-02 -5.296506e-04 9.999992e-01 -1.154865e-03 -2.839928e-02 2.066324e-03 1.155958e-03 9.999971e-01 8.586941e-01"
gt_trans = [ float(item) for item in gt_str.split(" ")]

gt_trans = megengine.tensor(gt_trans).reshape(3, 4)
gt_trans = F.concat([gt_trans, megengine.tensor([[0, 0, 0, 1]])], axis=0)
print(gt_trans)
gt_trans = F.broadcast_to(gt_trans, (batch, 4, 4))
target_point = F.matmul(gt_trans, source_points[:, :, None])   # [4, 4]    [4,4,1]
target_point = target_point[:, :, 0]


for i in range(1000):
    with gm:
        now_extrinsic = model(init_extrinsic, idx)
        batch = len(source_points)
        now_extrinsic = F.broadcast_to(now_extrinsic, (batch, 4, 4))
        predict_point = F.matmul(now_extrinsic, source_points[:, :, None])
        predict_point = predict_point[:, :, 0]
        loss  = ((target_point - predict_point)**2).sum()
        print("now loss: ", loss.item())
        gm.backward(loss)
        optimizer.step().clear_grad()

now_extrinsic = model(init_extrinsic, idx)
print(now_extrinsic)