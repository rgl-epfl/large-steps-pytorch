import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.main import optimize_shape
from scripts.constants import *
from scripts.io_ply import write_ply
from largesteps.optimize import AdamUniform
import torch

output_dir = os.path.join(OUTPUT_DIR, os.path.basename(os.path.dirname(__file__)))
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

folder = os.path.join(SCENES_DIR, "bunny_viewpoints")
num_viewpoints = [1, 2, 4, 9, 16, 25, 49]
pd.DataFrame(data=num_viewpoints).to_csv(os.path.join(output_dir, "viewpoints.csv"))

params = {
    "loss": "l1",
    "step_size": 1e-2,
    "boost" : 3,
    "alpha": 0.95,
    }

steps = [[5240, 4470, 3350, 2030, 1370, 930, 510],
         [6620, 5580, 3900, 2220, 1440, 960, 510]]
opt = [AdamUniform, torch.optim.Adam]
reg = [0, 2.1]

for i, n in enumerate(num_viewpoints):
    scene_name = f"bunny_{n:02d}"
    filename = os.path.join(folder, scene_name, scene_name + ".xml")
    for j, method in enumerate(["smooth", "reg"]):
        torch.cuda.empty_cache()
        params["smooth"] = (reg[j]==0)
        params["reg"] = reg[j]
        params["optimizer"] = opt[j]
        params["steps"] = steps[j][i]

        out = optimize_shape(filename, params)
        v = out["vert_steps"][-1] + out["tr_steps"][-1]
        f = out["f"][-1]
        write_ply(os.path.join(output_dir, f"res_{n:03d}_{method}.ply"), v, f)
        pd.DataFrame(data=out["losses"], columns=["im_loss", "reg_loss"]).to_csv(os.path.join(output_dir, f"loss_{n:03d}_{method}.csv"))

# Write target shape
write_ply(os.path.join(output_dir, "ref.ply"), out["v_ref"], out["f_ref"])
