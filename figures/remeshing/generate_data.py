import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../python/'))
from main import optimize_shape
from constants import *
from optimize import AdamUniform
from torch.optim import Adam
from io_ply import write_ply

output_dir = os.path.join(OUTPUT_DIR, os.path.basename(os.path.dirname(__file__)))
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

folder = SCENES_DIR
scene_name = "cranium"
filename = os.path.join(folder, scene_name, scene_name + ".xml")

params = {
    "time": 1,
    "boost" : 3,
    "loss": "l1",
    "smooth" : True,
    "lambda": 19,
    }

remesh = [-1, -1, 750, 0]
masses = []
verts = []
faces = []
for i, method in enumerate(["reg", "base", "remesh_middle", "remesh_start"]):
    if method == "reg":
        params["smooth"] = False
        params["optimizer"] = Adam
        params["reg"] = 0.16
        params["step_size"] = 1e-2
    else:
        params["smooth"] = True
        params["optimizer"] = AdamUniform
        params["reg"] = 0
        params["step_size"] = 1e-1
    params["remesh"] = remesh[i]
    out = optimize_shape(filename, params)
    v = out["vert_steps"][-1] + out["tr_steps"][-1]
    f = out["f"][-1]
    write_ply(os.path.join(output_dir, f"res_{method}.ply"), v, f)
