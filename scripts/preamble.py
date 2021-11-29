import numpy as np
import subprocess
import sys
import os

from scripts.constants import *

import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"""\usepackage{libertine}
\usepackage{amsmath}"""
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set()
sns.set(font_scale=1.1)

fontsize = 24
basename = os.path.join(OUTPUT_DIR, os.path.basename(os.getcwd()))

def blender_render(mesh_path, out_path, collection, viewpoint, res=100, area=False, ours=False, baseline=False, wireframe=False, t=0.008):
    """
    Render a mesh with blender. This method calls a blender python script using
    subprocess and an associated blender file containing a readily available
    rendering setup.

    Parameters
    ----------

    mesh_path : str
        Path to the model to render. PLY and OBJ files are supported.
    out_path : str
        Path to the folder to which the rendering will be saved. It will be named [mesh_name]_[wireframe/smooth].png
    collection: str
        Name of the collection in the blend file from which to choose a camera.
    viewpoint : int
        Index of the camera in the given collection.
    res : int
        Percentage of the full resolution in blender (default 100%)
    area : bool
        Vizualize vertex area as vertex colors (assumes vertex colors have been precomputed)
    ours : bool
        Wether this mesh is generated using our method or not.
    baseline : bool
        Wether this mesh is generated using a baseline or not.
    wireframe : bool
        Render the model with or without wireframe.
    t : float
        Wireframe thickness
    """
    args = [BLENDER_EXEC, "-b" , BLEND_SCENE, "--python",
            os.path.join(os.path.dirname(__file__), "blender_render.py"), "--", "-i", mesh_path,
            "-o", out_path, "-c", f"{collection}", "-v", f"{viewpoint}", "-r", f"{res}", "-t", f"{t}"]
    if baseline:
        args.append("--baseline")
    elif ours:
        args.append("--ours")
    if not wireframe:
        args.append("-s")
    if area:
        args.append("--area")
    subprocess.run(args, check=True)
