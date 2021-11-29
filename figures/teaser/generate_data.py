import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.main import optimize_shape
from scripts.constants import *
from scripts.io_ply import write_ply
from largesteps.optimize import AdamUniform

output_dir = os.path.join(OUTPUT_DIR, os.path.basename(os.path.dirname(__file__)))
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

scene_name = "nefertiti"
filename = os.path.join(SCENES_DIR, scene_name, scene_name + ".xml")

params = {
    "time": 2,
    "boost": 3,
    "lambda": 49,
    "loss": "l1"
}

reg = [0,0,16,0]
lr = [1e-1, 1e-1, 1e-2, 1e-2]
opt = [AdamUniform, AdamUniform, torch.optim.Adam, torch.optim.Adam]
smooth = [True, True, False, False]
remesh = [-1,250, -1,-1]

for j, method in enumerate(["ours", "remesh", "reg", "naive"]):
    torch.cuda.empty_cache()
    params["reg"] = reg[j]
    params["step_size"] = lr[j]
    params["optimizer"] = opt[j]
    params["smooth"] = smooth[j]
    params["remesh"] = remesh[j]

    out = optimize_shape(filename, params)
    v = out["vert_steps"][-1] + out["tr_steps"][-1]
    f = out["f"][-1]
    write_ply(os.path.join(output_dir, f"res_{method}.ply"), v, f)

# Write the source mesh
v = out["vert_steps"][0] + out["tr_steps"][0]
f = out["f"][0]
write_ply(os.path.join(output_dir, f"res_base.ply"), v, f)

# Write the target mesh
write_ply(os.path.join(output_dir, "ref.ply"), out["v_ref"], out["f_ref"])
