import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.main import optimize_shape
from scripts.constants import *
from scripts.io_ply import write_ply
from largesteps.optimize import AdamUniform
from torch.optim import Adam

output_dir = os.path.join(OUTPUT_DIR, os.path.basename(os.path.dirname(__file__)))
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

folder = SCENES_DIR
scene_name = "cranium"
filename = os.path.join(folder, scene_name, scene_name + ".xml")

params = {
    "boost" : 3,
    "step_size": 1e-2,
    "loss": "l1",
    "smooth" : True,
    "alpha": 0.95,
    }

remesh = [-1, -1, 750, 0]
steps = [1890, 1800, 1630, 1500]
masses = []
verts = []
faces = []
for i, method in enumerate(["reg", "base", "remesh_middle", "remesh_start"]):
    if method == "reg":
        params["smooth"] = False
        params["optimizer"] = Adam
        params["reg"] = 0.16
    else:
        params["smooth"] = True
        params["optimizer"] = AdamUniform
        params["reg"] = 0

    params["steps"] = steps[i]
    params["remesh"] = remesh[i]
    out = optimize_shape(filename, params)
    v = out["vert_steps"][-1] + out["tr_steps"][-1]
    f = out["f"][-1]
    write_ply(os.path.join(output_dir, f"res_{method}.ply"), v, f)
