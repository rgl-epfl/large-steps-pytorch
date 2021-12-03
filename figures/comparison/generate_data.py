import sys
import os
import numpy as np
import pandas as pd
import torch
from tqdm import trange

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.main import optimize_shape
from scripts.constants import *
from scripts.io_ply import write_ply
from largesteps.optimize import AdamUniform

try:
    from igl import hausdorff
except ModuleNotFoundError:
    print("WARNING: could not import libigl. The Hausdorff distances will not be computed. Please install libigl if you want to compute them.")

output_dir = os.path.join(OUTPUT_DIR, os.path.basename(os.path.dirname(__file__)))

scenes = ["suzanne", "cranium", "bob", "bunny", "tshirt", "planck"]
step_sizes = [2e-3, 5e-3, 3e-3, 1e-2, 3e-3, 3e-3]
durations = [0.5, 1, 0.5, 0.75, 0.2, 0.5]
regs = [2.8, 0.21, 0.67, 3.8, 12, 3.8]
regs_bi = [3.8, 0.16, 0.37, 2.1, 12, 5]

params = {
    "boost": 3,
    "loss": "l1",
    "alpha": 0.95,
}

for i, scene in enumerate(scenes):
    filename = os.path.join(SCENES_DIR, scene, f"{scene}.xml")
    output = os.path.join(output_dir, scene)
    if not os.path.isdir(output):
        os.makedirs(output)
    params["time"] = durations[i]
    for j, method in enumerate(["smooth", "reg", "bi"]):
        if j == 0:
            params["reg"] = 0
            params["smooth"] = True
            params["optimizer"] = AdamUniform
            params["step_size"] = step_sizes[i]
        else:
            if j==1:
                params["reg"] = regs[i]
                params["bilaplacian"] = False
            else:
                params["reg"] = regs_bi[i]
                params["bilaplacian"] = True
            params["smooth"] = False
            params["optimizer"] = torch.optim.Adam
            params["step_size"] = 1e-2

        torch.cuda.empty_cache()
        out = optimize_shape(filename, params)
        # Write result
        v = out["vert_steps"][-1] + out["tr_steps"][-1]
        f = out["f"][-1]
        write_ply(os.path.join(output, f"res_{method}.ply"), v, f)

        # Write base mesh, reference shape and images
        if j == 0:
            v = out["vert_steps"][0] + out["tr_steps"][0]
            f = out["f"][0]
            write_ply(os.path.join(output, f"base.ply"), v, f)

            # Write the reference shape
            write_ply(os.path.join(output, "ref.ply"), out["v_ref"], out["f_ref"])

        losses = np.zeros((out["losses"].shape[0], 3))
        losses[:,:2] = out["losses"]
        if "hausdorff" in dir():
            # Compute the hausdorff distance
            vb = out["v_ref"]
            fb = out["f_ref"]
            fa = out["f"][0]
            verts = (np.array(out["vert_steps"]) + np.array(out["tr_steps"]))[1::10]
            d_hausdorff = np.zeros((verts.shape[0]))
            for it in trange(verts.shape[0]):
                d_hausdorff[it] = (hausdorff(verts[it], fa, vb, fb) + hausdorff(vb, fb, verts[it], fa))

            losses[1::10,2] = d_hausdorff

        # Write the losses
        pd.DataFrame(data=losses, columns=["im_loss", "reg_loss", "hausdorff"]).to_csv(os.path.join(output, f"loss_{method}.csv"))
