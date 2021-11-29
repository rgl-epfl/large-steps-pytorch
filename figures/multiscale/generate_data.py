import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.main import optimize_shape
from scripts.constants import *
from scripts.io_ply import write_ply

output_dir = os.path.join(OUTPUT_DIR, os.path.basename(os.path.dirname(__file__)))
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

folder = SCENES_DIR
scene = "dragon"
filename = os.path.join(folder, scene, f"{scene}.xml")
remesh_steps = [500, 1500, 3000, 4500, 7000, 10000, 12000, 14000]

params = {
    "steps": 16000,
    "step_size" : 1e-1,
    "loss": "l1",
    "boost" : 3,
    "lambda": 19,
    "remesh": remesh_steps.copy(),
    }

out = optimize_shape(filename, params)

all_steps = [0, *remesh_steps, -1]
N = len((all_steps)) - 2
pd.DataFrame(data=all_steps).to_csv(os.path.join(output_dir, "remesh_steps.csv"))
for i, step in enumerate(all_steps):
    v = out["vert_steps"][step] + out["tr_steps"][step]
    f = out["f"][min(i, N)]
    write_ply(os.path.join(output_dir, f"res_{i:02d}.ply"), v, f)

v = out["vert_steps"][-1] + out["tr_steps"][-1]
f = out["f"][-1]
write_ply(os.path.join(output_dir, f"res_{i+1:02d}.ply"), v, f)
