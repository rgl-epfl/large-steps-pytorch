import torch
from mitsuba.core import Int
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

def matrix_to_torch(M):
    """
    Manually convert an enoki Matrix4f to a PyTorch tensor.
    """
    m_torch = torch.zeros((4,4), dtype=torch.float, device='cuda')
    for i in range(4):
        for j in range(4):
            m_torch[i,j] = M[i,j][0]
    return m_torch

def load_shape(params, id_):
    """
    Load the enoki arrays making up a shape and convert them to PyTorch tensors.

    Loads the vertex positions, vertex normals and triangle faces.

    Parameters
    ----------
    params : mitsuba.python.util.SceneParameters
        The scene parameters
    id_ : str
        The id of the shape to load

    Returns
    -------
    The corresponding PyTorch tensors.
    """
    # Vertex positions
    v = params[f"{id_}.vertex_positions"]
    vert_torch = v.torch().view((-1,3))
    # Vertex normals
    n = params[f"{id_}.vertex_normals"]
    norm_torch = n.torch().view((-1,3))
    # Triangle faces
    f = params[f"{id_}.faces"]
    faces_torch = Int(f).torch().view((-1,3))
    return vert_torch, norm_torch, faces_torch

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
    proj = torch.tensor([[-1.0 / np.tan(fov_rad / 2.0), 0, 0, 0],
                      [0, ar / np.tan(fov_rad / 2.0), 0, 0],
                      [0, 0, -(near + far) / (near-far), 2 * far * near / (near-far)],
                      [0, 0, 1, 0]], dtype=torch.float32, device='cuda')
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

def writePLY(filename, V, F, VC = None):
    """
    Write a mesh as PLY with vertex colors.

    Parameters
    ----------
    filename : str
        Path to which to save the mesh.
    V : numpy.ndarray
        Vertex positions.
    F : numpy.ndarray
        Faces.
    VC : numpy.ndarray
        Vertex colors (optional).
    """
    color = (VC*255).astype(np.uint8)
    f = open(filename, 'w')
    # headers
    string = 'ply\n'
    string = string + 'format ascii 1.0\n'
    string = string + 'element vertex ' + str(V.shape[0]) + '\n'
    string = string + 'property double x\n'
    string = string + 'property double y\n'
    string = string + 'property double z\n'
    if (VC is not None and VC.shape[0] == V.shape[0]):
        string = string + 'property uchar red\n'
        string = string + 'property uchar green\n'
        string = string + 'property uchar blue\n'
        string = string + 'property uchar alpha\n'

    # end of header
    string = string + 'element face ' + str(F.shape[0]) + '\n'
    string = string + 'property list int int vertex_indices\n'
    string = string + 'end_header\n'
    f.write(string)
    # write vertices
    for ii in range(V.shape[0]):
        string = '%f %f %f ' % (V[ii,0], V[ii,1], V[ii,2])
        if (VC is not None and VC.shape[0] == V.shape[0]):
            string = string + '%03d %03d %03d %03d\n' % (color[ii,0], color[ii,1], color[ii,2], 255)
        else:
            string = string + '\n'
        f.write(string)
    for ii in range(F.shape[0]):
        string = '%d %d %d %d\n' % (3, F[ii,0], F[ii,1], F[ii,2])
        f.write(string)
    f.close()

def safe_acos(x):
    return torch.acos(x.clamp(min=-1, max=1))
