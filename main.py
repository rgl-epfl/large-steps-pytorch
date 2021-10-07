import torch
import mitsuba
mitsuba.set_variant("cuda_ad_rgb")
from mitsuba.core import Float, Thread
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse
import time
import os
from tqdm import tqdm

from optimize import AdamUniform
from utils import load_shape
from render import NVDRenderer
from geometry import compute_matrix, remove_duplicates, laplacian_cot, laplacian_uniform, compute_face_normals, compute_vertex_normals
from parameterize import to_differential, from_differential
from solvers import CholeskySolver

def optimize_shape(filepath, params):
    """
    Optimize a shape given a scene.

    This will expect a Mitsuba scene as input containing

    Parameters
    ----------
    filepath : str
        Path to the XML file of the scene to optimize.
    params : dict
        Dictionary containing all optimization parameters.
    """
    opt_time = params.get("time", -1) # Optimization time (in minutes)
    steps = params.get("steps", 100) # Number of optimization steps (ignored if time > 0)
    lr = params.get("lr", 0.01) # Step size
    boost = params.get("boost", 1) # Gradient boost used in nvdiffrast
    smooth = params.get("smooth", True) # Use our method or not
    shading = params.get("shading", True) # Use shading, otherwise render silhouettes
    reg = params.get("reg", 0.0) # Regularization weight
    save = params.get("save", -1) # Save renderings every 'save' steps (if > 0)
    save_mesh = params.get("save_mesh", False) # Also save the mesh when saving
    OUTPUT_DIR = params.get("output", "/home/bnicolet/Documents/.optim") # Where to save the images/meshes
    cotan = params.get("cotan", False) # Use cotan laplacian, otherwise use the combinatorial one (more efficient)
    solver = params.get("solver", 'Cholesky') # Solver to use
    lambda_ = params.get("lambda", 1.0) # Hyperparameter lambda of our method, used to compute the matrix (I + lambda_*L)
    subdiv = params.get("subdiv", -1) # Time step(s) at which to remesh
    optimizer = params.get("optimizer", AdamUniform) # Which optimizer to use
    use_tr = params.get("use_tr", True) # Optimize a global translation at the same time
    loss_function = params.get("loss", "l2") # Which loss to use

    # Load the scene
    folder, scene_file = os.path.split(filepath)
    scene_name, ext = os.path.splitext(scene_file)
    assert ext == ".xml", f"Unexpected file type: '{ext}'"
    Thread.thread().file_resolver().prepend(folder)
    scene = load_file(filepath)
    scene_params = traverse(scene)

    # Load reference shape
    v_ref, n_ref, f_ref = load_shape(scene_params, 'mesh-target')
    # Load source shape
    v_src, n_src, f_src = load_shape(scene_params, 'mesh-source')
    # Remove duplicates. This is necessary to avoid seams of meshes to rip apart during the optimization
    v_unique, f_unique, duplicate_idx = remove_duplicates(v_src, f_src)

    # Initialize the renderer
    renderer = NVDRenderer(scene, shading=shading, boost=boost)

    # Render the reference images
    ref_imgs = renderer.render(v_ref, n_ref, f_ref)

    # Compute the laplacian for the regularization term
    if cotan:
        L = laplacian_cot(v_unique, f_unique)
    else:
        L = laplacian_uniform(v_unique, f_unique)

    # Initialize the optimized variables and the optimizer
    tr = torch.zeros((1,3), device='cuda', dtype=torch.float32)
    opt_params = []

    if use_tr:
        tr.requires_grad = True
        opt_params.append(tr)
    if smooth:
        # Compute the system matrix and parameterize
        M = compute_matrix(v_unique, f_unique, lambda_)# / (1+lambda_)
        u_unique = to_differential(M, v_unique)#/(1+lambda_)
        u_unique.requires_grad = True
        opt_params.append(u_unique)
    else:
        v_unique.requires_grad = True
        opt_params.append(v_unique)

    opt = optimizer([u_unique], lr=lr)
    # TODO: this is an ugly workaround to reproduce the results from the paper, we shouldn't do this
    opt_tr = optimizer([tr], lr=lr/(1+lambda_))

    # Set values for time and step count
    if opt_time > 0:
        steps = -1
    it = 0
    t0 = time.perf_counter()
    t = t0
    opt_time *= 60

    # Dictionary that is returned in the end, contains useful information for debug/analysis
    result_dict = {"vert_steps": [], "tr_steps": [], "f": [f_src.cpu().numpy().copy()],
                "losses": [], "im_ref": ref_imgs.cpu().numpy().copy()}
    #solver = CholeskySolver(M)
    # Optimization loop
    with tqdm(total=max(steps, opt_time), ncols=100, bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
        while it < steps or (t-t0) < opt_time:

            #TODO: remeshing
            # Get cartesian coordinates
            if smooth:
                v_unique = from_differential(M, u_unique)

            #TODO: save, timing
            # Get the version of the mesh with the duplicates
            v_opt = v_unique[duplicate_idx]
            result_dict["vert_steps"].append(v_opt.detach().cpu().numpy().copy())
            result_dict["tr_steps"].append(tr.detach().cpu().numpy().copy())
            # Recompute vertex normals
            face_normals = compute_face_normals(v_unique, f_unique)
            n_unique = compute_vertex_normals(v_unique, f_unique, face_normals)
            n_opt = n_unique[duplicate_idx]

            # Render images
            opt_imgs = renderer.render(tr + v_opt, n_opt, f_src)

            # Compute image loss
            if loss_function == "l1":
                loss = (opt_imgs - ref_imgs).abs().mean()
            elif loss_function == "l2":
                loss = (opt_imgs - ref_imgs).square().mean()

            # Add regularization
            if reg > 0:
                loss = loss + reg * (L@v_unique).square().mean() # TODO: add bilaplacian baseline

            result_dict["losses"].append(loss.detach().cpu().numpy().copy())
            # Backpropagate
            opt.zero_grad()
            loss.backward()
            # Update parameters
            #u_unique.grad *= (1+lambda_)**2
            opt.step()
            opt_tr.step()

            it += 1
            t = time.perf_counter()
            if steps > -1:
                pbar.update(1)
            else:
                pbar.update(min(opt_time, (t-t0)) - pbar.n)

    return result_dict

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize the geometry of a mesh.")
    parser.add_argument("scene", type=os.path.abspath, help="Path to the folder containing the 'scene.xml' file.")
    parser.add_argument("--name", type=str)
    parser.add_argument("--time", type=float, default=-1) #optimization time (in minutes)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--boost", type=float, default=1)
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--reg", type=float, default=0.0)
    parser.add_argument("--output", type=os.path.abspath, default="/home/bnicolet/Documents/.optim")
    parser.add_argument("--influence", type=float, default=1.0)
    parser.add_argument("--subdiv", type=int, default=-1)
    parser.add_argument("--adam", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--loss", type=str, default="l1")
    parser.add_argument("--bilaplacian", action="store_true")
    parser.add_argument("--precond", action="store_true")
    parser.add_argument("--deg", type=int, default=2)

    args = parser.parse_args()
    params = vars(args)
    if args.adam:
        params["optimizer"] = torch.optim.Adam
    else:
        params["optimizer"] = AdamUniform
    out = optimize_shape(args.scene, params)
    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    loss = out["losses"][:,0]
    np.save(os.path.join(args.output, f"{args.name}_loss.npy"), loss)
    # Write result
    v = out["vert_steps"][-1] + out["tr_steps"][-1]
    f = out["f"][-1]
    write_obj(os.path.join(args.output, f"{args.name}.obj"), v, f)
