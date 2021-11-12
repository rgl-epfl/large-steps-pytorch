import torch
import numpy as np
from cupy._core.dlpack import toDlpack
from cupy._core.dlpack import fromDlpack
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

def torch_to_cupy(x):
    """
    Convert a PyTorch tensor to a CuPy array.
    """
    return fromDlpack(to_dlpack(x))

def cupy_to_torch(x):
    """
    Convert a CuPy array to a PyTorch tensor.
    """
    return from_dlpack(toDlpack(x))

def persp_proj(fov_x=45, ar=1, near=0.1, far=100):
    """
    Build a perspective projection matrix.

    Parameters
    ----------
    fov_x : float
        Horizontal field of view (in degrees).
    ar : float
        Aspect ratio (w/h).
    near : float
        Depth of the near plane relative to the camera.
    far : float
        Depth of the far plane relative to the camera.
    """
    fov_rad = np.deg2rad(fov_x)
    proj_mat = np.array([[-1.0 / np.tan(fov_rad / 2.0), 0, 0, 0],
                      [0, np.float32(ar) / np.tan(fov_rad / 2.0), 0, 0],
                      [0, 0, -(near + far) / (near-far), 2 * far * near / (near-far)],
                      [0, 0, 1, 0]])
    x = torch.tensor([[1,2,3,4]], device='cuda')
    proj = torch.tensor(proj_mat, device='cuda', dtype=torch.float32)
    return proj

def tonemap(image, ev=0.0):
    """
    Perform gamma correction on the image with EV adjusting

    Parameters
    ----------
    image : torch.Tensor
        Image to gamma correct
    ev : float
        EV offset to apply (default 0.0)
    """
    adjusted_img = image * 2**ev
    return torch.where(adjusted_img < 0.0031308, 12.92 * adjusted_img, 1.055*adjusted_img.pow(1/2.4)-0.055)

def safe_acos(x):
    return torch.acos(x.clamp(min=-1, max=1))
