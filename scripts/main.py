import torch
import time
import os
from tqdm import tqdm
import numpy as np
import sys

from largesteps.optimize import AdamUniform
from largesteps.geometry import compute_matrix, laplacian_uniform
from largesteps.parameterize import to_differential, from_differential
from scripts.render import NVDRenderer
from scripts.load_xml import load_scene
from scripts.constants import REMESH_DIR
from scripts.geometry import remove_duplicates, compute_face_normals, compute_vertex_normals, average_edge_length
sys.path.append(REMESH_DIR)
from pyremesh import remesh_botsch

def optimize_shape(filepath, params):
    """
    Optimize a shape given a scene.

    This will expect a Mitsuba scene as input containing the cameras, envmap and
    source and target models.

    Parameters
    ----------
    filepath : str Path to the XML file of the scene to optimize. params : dict
        Dictionary containing all optimization parameters.
    """
    opt_time = params.get("time", -1) # Optimization time (in minutes)
    steps = params.get("steps", 100) # Number of optimization steps (ignored if time > 0)
    step_size = params.get("step_size", 0.01) # Step size
    boost = params.get("boost", 1) # Gradient boost used in nvdiffrast
    smooth = params.get("smooth", True) # Use our method or not
    shading = params.get("shading", True) # Use shading, otherwise render silhouettes
    reg = params.get("reg", 0.0) # Regularization weight
    solver = params.get("solver", 'Cholesky') # Solver to use
    lambda_ = params.get("lambda", 1.0) # Hyperparameter lambda of our method, used to compute the parameterization matrix as (I + lambda_ * L)
    alpha = params.get("alpha", None) # Alternative hyperparameter, used to compute the parameterization matrix as ((1-alpha) * I + alpha * L)
    remesh = params.get("remesh", -1) # Time step(s) at which to remesh
    optimizer = params.get("optimizer", AdamUniform) # Which optimizer to use
    use_tr = params.get("use_tr", True) # Optimize a global translation at the same time
    loss_function = params.get("loss", "l2") # Which loss to use
    bilaplacian = params.get("bilaplacian", True) # Use the bilaplacian or the laplacian regularization loss

    # Load the scene
    scene_params = load_scene(filepath)

    # Load reference shape
    v_ref = scene_params["mesh-target"]["vertices"]
    f_ref = scene_params["mesh-target"]["faces"]
    if "normals" in scene_params["mesh-target"].keys():
        n_ref = scene_params["mesh-target"]["normals"]
    else:
        face_normals = compute_face_normals(v_ref, f_ref)
        n_ref = compute_vertex_normals(v_ref, f_ref, face_normals)

    # Load source shape
    v_src = scene_params["mesh-source"]["vertices"]
    f_src = scene_params["mesh-source"]["faces"]
    # Remove duplicates. This is necessary to avoid seams of meshes to rip apart during the optimization
    v_unique, f_unique, duplicate_idx = remove_duplicates(v_src, f_src)

    # Initialize the renderer
    renderer = NVDRenderer(scene_params, shading=shading, boost=boost)

    # Render the reference images
    ref_imgs = renderer.render(v_ref, n_ref, f_ref)

    # Compute the laplacian for the regularization term
    L = laplacian_uniform(v_unique, f_unique)

    # Initialize the optimized variables and the optimizer
    tr = torch.zeros((1,3), device='cuda', dtype=torch.float32)

    if smooth:
        # Compute the system matrix and parameterize
        M = compute_matrix(v_unique, f_unique, lambda_=lambda_, alpha=alpha)
        u_unique = to_differential(M, v_unique)

    def initialize_optimizer(u, v, tr, step_size):
        """
        Initialize the optimizer

        Parameters
        ----------
        - u : torch.Tensor or None
            Parameterized coordinates to optimize if not None
        - v : torch.Tensor
            Cartesian coordinates to optimize if u is None
        - tr : torch.Tensor
            Global translation to optimize if not None
        - step_size : float
            Step size

        Returns
        -------
        a torch.optim.Optimizer containing the tensors to optimize.
        """
        opt_params = []
        if tr is not None:
            tr.requires_grad = True
            opt_params.append(tr)
        if u is not None:
            u.requires_grad = True
            opt_params.append(u)
        else:
            v.requires_grad = True
            opt_params.append(v)

        return optimizer(opt_params, lr=step_size)

    opt = initialize_optimizer(u_unique if smooth else None, v_unique, tr if use_tr else None, step_size)

    # Set values for time and step count
    if opt_time > 0:
        steps = -1
    it = 0
    t0 = time.perf_counter()
    t = t0
    opt_time *= 60

    # Dictionary that is returned in the end, contains useful information for debug/analysis
    result_dict = {"vert_steps": [], "tr_steps": [], "f": [f_src.cpu().numpy().copy()],
                "losses": [], "im_ref": ref_imgs.cpu().numpy().copy(), "im":[],
                "v_ref": v_ref.cpu().numpy().copy(), "f_ref": f_ref.cpu().numpy().copy()}

    if type(remesh) == list:
        remesh_it = remesh.pop(0)
    else:
        remesh_it = remesh


    # Optimization loop
    with tqdm(total=max(steps, opt_time), ncols=100, bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
        while it < steps or (t-t0) < opt_time:
            if it == remesh_it:
                # Remesh
                with torch.no_grad():
                    if smooth:
                        v_unique = from_differential(M, u_unique, solver)

                    v_cpu = v_unique.cpu().numpy()
                    f_cpu = f_unique.cpu().numpy()
                    # Target edge length
                    h = (average_edge_length(v_unique, f_unique)).cpu().numpy()*0.5

                    # Run 5 iterations of the Botsch-Kobbelt remeshing algorithm
                    v_new, f_new = remesh_botsch(v_cpu.astype(np.double), f_cpu.astype(np.int32), 5, h, True)

                    v_src = torch.from_numpy(v_new).cuda().float().contiguous()
                    f_src = torch.from_numpy(f_new).cuda().contiguous()

                    v_unique, f_unique, duplicate_idx = remove_duplicates(v_src, f_src)
                    result_dict["f"].append(f_new)
                    # Recompute laplacian
                    L = laplacian_uniform(v_unique, f_unique)

                    if smooth:
                        # Compute the system matrix and parameterize
                        M = compute_matrix(v_unique, f_unique, lambda_=lambda_, alpha=alpha)
                        u_unique = to_differential(M, v_unique)

                    step_size *= 0.8
                    opt = initialize_optimizer(u_unique if smooth else None, v_unique, tr if use_tr else None, step_size)

                # Get next remesh iteration if any
                if type(remesh) == list and len(remesh) > 0:
                    remesh_it = remesh.pop(0)

            # Get cartesian coordinates
            if smooth:
                v_unique = from_differential(M, u_unique, solver)

            # Get the version of the mesh with the duplicates
            v_opt = v_unique[duplicate_idx]
            # Recompute vertex normals
            face_normals = compute_face_normals(v_unique, f_unique)
            n_unique = compute_vertex_normals(v_unique, f_unique, face_normals)
            n_opt = n_unique[duplicate_idx]

            # Render images
            opt_imgs = renderer.render(tr + v_opt, n_opt, f_src)

            # Compute image loss
            if loss_function == "l1":
                im_loss = (opt_imgs - ref_imgs).abs().mean()
            elif loss_function == "l2":
                im_loss = (opt_imgs - ref_imgs).square().mean()

            # Add regularization
            if bilaplacian:
                reg_loss = (L@v_unique).square().mean()
            else:
                reg_loss = (v_unique * (L @v_unique)).mean()

            loss = im_loss + reg * reg_loss

            # Record optimization state for later processing
            result_dict["losses"].append((im_loss.detach().cpu().numpy().copy(), (L@v_unique.detach()).square().mean().cpu().numpy().copy()))
            result_dict["vert_steps"].append(v_opt.detach().cpu().numpy().copy())
            result_dict["tr_steps"].append(tr.detach().cpu().numpy().copy())

            # Backpropagate
            opt.zero_grad()
            loss.backward()
            # Update parameters
            opt.step()

            it += 1
            t = time.perf_counter()
            if steps > -1:
                pbar.update(1)
            else:
                pbar.update(min(opt_time, (t-t0)) - pbar.n)

    result_dict["losses"] = np.array(result_dict["losses"])
    return result_dict
