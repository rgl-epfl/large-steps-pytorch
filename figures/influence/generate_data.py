import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.main import *
from scripts.constants import *
from scripts.io_ply import write_ply
sys.path.append(REMESH_DIR)

folder = SCENES_DIR
scene_name = "suzanne"
filename = os.path.join(folder, scene_name, scene_name + ".xml")

output_dir = os.path.join(OUTPUT_DIR, os.path.basename(os.path.dirname(__file__)))
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

params = {
    "time": 2,
    "loss": "l1",
    "boost" : 3,
    "optimizer": AdamUniform
    }

lambdas = [0., 0.3333, 1., 3., 19., 49., 99., 999.]
step_sizes = [0.001, 0.0013, 0.002, 0.004, 0.02, 0.05, 0.1, 1.]

df = pd.DataFrame(data={"lambda": lambdas})
df.to_csv(os.path.join(output_dir, "lambdas.csv"))
for i, lambda_ in enumerate(lambdas):
    params["lambda"] = lambda_
    params["step_size"] = step_sizes[i]

    out = optimize_shape(filename, params)
    v = out["vert_steps"][-1] + out["tr_steps"][-1]
    print(len(out["vert_steps"]))
    f = out["f"][-1]
    # Write the resulting mesh
    write_ply(os.path.join(output_dir, f"res_{i:02d}.ply"), v, f)

# Write target shape
write_ply(os.path.join(output_dir, "ref.ply"), out["v_ref"], out["f_ref"])
