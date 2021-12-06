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
    "steps": 4300,
    "loss": "l1",
    "boost" : 3,
    "step_size": 1e-3,
    "optimizer": AdamUniform
    }

alphas = [0.0, 0.25, 0.5, 0.75, 0.95, 0.98, 0.99, 0.999]

df = pd.DataFrame(data={"alpha": alphas})
df.to_csv(os.path.join(output_dir, "alphas.csv"))
for i, alpha in enumerate(alphas):
    params["alpha"] = alpha

    out = optimize_shape(filename, params)
    v = out["vert_steps"][-1] + out["tr_steps"][-1]
    f = out["f"][-1]
    # Write the resulting mesh
    write_ply(os.path.join(output_dir, f"res_{i:02d}.ply"), v, f)

# Write target shape
write_ply(os.path.join(output_dir, "ref.ply"), out["v_ref"], out["f_ref"])
