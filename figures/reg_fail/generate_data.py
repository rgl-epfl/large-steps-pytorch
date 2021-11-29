import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.main import optimize_shape
from scripts.constants import *
from scripts.io_ply import write_ply
from largesteps.optimize import AdamUniform
from torch.optim import Adam

output_dir = os.path.join(OUTPUT_DIR, os.path.basename(os.path.dirname(__file__)))
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

scene_name = "reg_fail"
filename = os.path.join(SCENES_DIR, scene_name, scene_name + ".xml")

reg_weights = [1, 400, 1e3]
df = pd.DataFrame(data={"weight":reg_weights})
df.to_csv(os.path.join(output_dir, f"weights.csv"))

save_steps = [0, 500, 2500, 7500, 15000, 25000]
df = pd.DataFrame(data={"save_steps": save_steps})
df.to_csv(os.path.join(output_dir, f"save_steps.csv"))

params = {
    "steps" : 25001,
   "step_size" : 5e-3,
    "shading": False,
    "boost" : 3,
    "smooth" : True,
    "lambda": 99,
    "loss": "l2",
    "use_tr": False,
    "optimizer": AdamUniform,
    "reg": 0
    }

# Optimize with our method
out = optimize_shape(filename, params)
for step in save_steps:
    v = out["vert_steps"][step] + out["tr_steps"][step]
    f = out["f"][-1]
    write_ply(os.path.join(output_dir, f"smooth_{step:05d}.ply"), v, f)

# Write out the loss
data = {
    'im_loss': out["losses"][:,0],
    'reg_loss': out["losses"][:,1]
}
df = pd.DataFrame(data=data)
df.to_csv(os.path.join(output_dir, f"loss_smooth.csv"))

params["smooth"] = False
params["optimizer"] = Adam
params["step_size"] = 1e-4
for i, w in enumerate(reg_weights):
    params["reg"] = w
    # Optimize with regularization
    out = optimize_shape(filename, params)
    for step in save_steps:
        v = out["vert_steps"][step] + out["tr_steps"][step]
        f = out["f"][-1]
        write_ply(os.path.join(output_dir, f"reg_{i}_{step:05d}.ply"), v, f)

    data = {
        'im_loss': out["losses"][:,0],
        'reg_loss': out["losses"][:,1]
    }
    df = pd.DataFrame(data=data)
    df.to_csv(os.path.join(output_dir, f"loss_reg_{i}.csv"))
