import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import megengine
import megengine.functional as F
from megengine.autodiff import GradManager
import megengine.module as M
import megengine.optimizer as optim
import numpy as np


class MyModel(M.Module):
     def __init__(self):
         super().__init__()
         self.fc = M.Linear(50, 4)

     def forward(self, input):
         x = self.fc(input)
         return x

model = MyModel()
gm = GradManager().attach(model.parameters())
optimizer = optim.Adam(model.parameters(), lr=0.01)

input =  F.zeros((100, 50))

with gm:
    output = model(input) # (100, 4)
    output = output.reshape(2, 50, 4)

    rgb = output[:, :, :3] # (2, 50, 3)
    weights = output[:, :, 3:4] # (2, 50, 1)
    rgb = F.sum(weights * rgb, axis=-2) # (2, 3)

    # annotation this line is ok
    some_info_of_weights = F.sum(weights, axis=-2)

    target = F.ones((2, 3))
    loss = F.abs(target - rgb).sum()
    gm.backward(loss)
    optimizer.step().clear_grad()
