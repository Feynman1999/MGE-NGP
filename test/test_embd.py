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

num_views = 5
model = megengine.module.Embedding(num_views, 6) # [5,6]
print(model.weight)
idx = megengine.tensor([[0,1,3],[1,0,2]], dtype=np.int32) # [2,3]
print(model(idx))
print(model(idx).shape)