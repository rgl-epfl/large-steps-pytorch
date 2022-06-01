import torch
import numpy as np
import nvdiffrast.torch as dr

class SphericalHarmonics:
    """
    Environment map approximation using spherical harmonics.

    This class implements the spherical harmonics lighting model of [Ramamoorthi
    and Hanrahan 2001], that approximates diffuse lighting by an environment map.
    """

    def __init__(self, envmap):
        """
        Precompute the coefficients given an envmap.

        Parameters
        ----------
        envmap : torch.Tensor
            The environment map to approximate.
        """
        h,w = envmap.shape[:2]

        # Compute the grid of theta, phi values
        theta = (torch.linspace(0, np.pi, h, device='cuda')).repeat(w, 1).t()
        phi = (torch.linspace(3*np.pi, np.pi, w, device='cuda')).repeat(h,1)

        # Compute the value of sin(theta) once
        sin_theta = torch.sin(theta)
        # Compute x,y,z
        # This differs from the original formulation as here the up axis is Y
        x = sin_theta * torch.cos(phi)
        z = -sin_theta * torch.sin(phi)
        y = torch.cos(theta)

        # Compute the polynomials
        Y_0 = 0.282095
        # The following are indexed so that using Y_n[-p]...Y_n[p] gives the proper polynomials
        Y_1 = [
            0.488603 * z,
            0.488603 * x,
            0.488603 * y
            ]
        Y_2 = [
            0.315392 * (3*z.square() - 1),
            1.092548 * x*z,
            0.546274 * (x.square() - y.square()),
            1.092548 * x*y,
            1.092548 * y*z
        ]
        import matplotlib.pyplot as plt
        area = w*h
        radiance = envmap[..., :3]
        dt_dp = 2.0 * np.pi**2 / area

        # Compute the L coefficients
        L = [ [(radiance * Y_0 * (sin_theta)[..., None] * dt_dp).sum(dim=(0,1))],
            [(radiance * (y * sin_theta)[..., None] * dt_dp).sum(dim=(0,1)) for y in Y_1],
            [(radiance * (y * sin_theta)[..., None] * dt_dp).sum(dim=(0,1)) for y in Y_2]]

        # Compute the R,G and B matrices
        c1 = 0.429043
        c2 = 0.511664
        c3 = 0.743125
        c4 = 0.886227
        c5 = 0.247708

        self.M = torch.stack([
            torch.stack([ c1 * L[2][2] , c1 * L[2][-2], c1 * L[2][1] , c2 * L[1][1]           ]),
            torch.stack([ c1 * L[2][-2], -c1 * L[2][2], c1 * L[2][-1], c2 * L[1][-1]          ]),
            torch.stack([ c1 * L[2][1] , c1 * L[2][-1], c3 * L[2][0] , c2 * L[1][0]           ]),
            torch.stack([ c2 * L[1][1] , c2 * L[1][-1], c2 * L[1][0] , c4 * L[0][0] - c5 * L[2][0]])
        ]).movedim(2,0)

    def eval(self, n):
        """
        Evaluate the shading using the precomputed coefficients.

        Parameters
        ----------
        n : torch.Tensor
            Array of normals at which to evaluate lighting.
        """
        normal_array = n.view((-1, 3))
        h_n = torch.nn.functional.pad(normal_array, (0,1), 'constant', 1.0)
        l = (h_n.t() * (self.M @ h_n.t())).sum(dim=1)
        return l.t().view(n.shape)

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

class NVDRenderer:
    """
    Renderer using nvdiffrast.


    This class encapsulates the nvdiffrast renderer [Laine et al 2020] to render
    objects given a number of viewpoints and rendering parameters.
    """
    def __init__(self, scene_params, shading=True, boost=1.0):
        """
        Initialize the renderer.

        Parameters
        ----------
        scene_params : dict
            The scene parameters. Contains the envmap and camera info.
        shading: bool
            Use shading in the renderings, otherwise render silhouettes. (default True)
        boost: float
            Factor by which to multiply shading-related gradients. (default 1.0)
        """
        # We assume all cameras have the same parameters (fov, clipping planes)
        near = scene_params["near_clip"]
        far = scene_params["far_clip"]
        self.fov_x = scene_params["fov"]
        w = scene_params["res_x"]
        h = scene_params["res_y"]
        self.res = (h,w)
        ar = w/h
        x = torch.tensor([[1,2,3,4]], device='cuda')
        self.proj_mat = persp_proj(self.fov_x, ar, near, far)

        # Construct the Model-View-Projection matrices
        self.view_mats = torch.stack(scene_params["view_mats"])
        self.mvps = self.proj_mat @ self.view_mats

        self.boost = boost
        self.shading = shading

        # Initialize rasterizing context
        self.glctx = dr.RasterizeGLContext()
        # Load the environment map
        w,h,_ = scene_params['envmap'].shape
        envmap = scene_params['envmap_scale'] * scene_params['envmap']
        # Precompute lighting
        self.sh = SphericalHarmonics(envmap)
        # Render background for all viewpoints once
        self.render_backgrounds(envmap)

    def render_backgrounds(self, envmap):
        """
        Precompute the background of each input viewpoint with the envmap.

        Params
        ------
        envmap : torch.Tensor
            The environment map used in the scene.
        """
        h,w = self.res
        pos_int = torch.arange(w*h, dtype = torch.int32, device='cuda')
        pos = 0.5 - torch.stack((pos_int % w, pos_int // w), dim=1) / torch.tensor((w,h), device='cuda')
        a = np.deg2rad(self.fov_x)/2
        r = w/h
        f = torch.tensor((2*np.tan(a),  2*np.tan(a)/r), device='cuda', dtype=torch.float32)
        rays = torch.cat((pos*f, torch.ones((w*h,1), device='cuda'), torch.zeros((w*h,1), device='cuda')), dim=1)
        rays_norm = (rays.transpose(0,1) / torch.norm(rays, dim=1)).transpose(0,1)
        rays_view = torch.matmul(rays_norm, self.view_mats.inverse().transpose(1,2)).reshape((self.view_mats.shape[0],h,w,-1))
        theta = torch.acos(rays_view[..., 1])
        phi = torch.atan2(rays_view[..., 0], rays_view[..., 2])
        envmap_uvs = torch.stack([0.75-phi/(2*np.pi), theta / np.pi], dim=-1)
        self.bgs = dr.texture(envmap[None, ...], envmap_uvs, filter_mode='linear').flip(1)
        self.bgs[..., -1] = 0 # Set alpha to 0

    def render(self, v, n, f):
        """
        Render the scene in a differentiable way.

        Parameters
        ----------
        v : torch.Tensor
            Vertex positions
        n : torch.Tensor
            Vertex normals
        f : torch.Tensor
            Model faces

        Returns
        -------
        result : torch.Tensor
            The array of renderings from all given viewpoints
        """
        v_hom = torch.nn.functional.pad(v, (0,1), 'constant', 1.0)
        v_ndc = torch.matmul(v_hom, self.mvps.transpose(1,2))
        rast = dr.rasterize(self.glctx, v_ndc, f, self.res)[0]
        if self.shading:
            v_cols = torch.zeros_like(v)

            # Sample envmap at each vertex using the SH approximation
            vert_light = self.sh.eval(n).contiguous()
            # Sample incoming radiance
            light = dr.interpolate(vert_light[None, ...], rast, f)[0]

            col = torch.cat((light / np.pi, torch.ones((*light.shape[:-1],1), device='cuda')), dim=-1)
            result = dr.antialias(torch.where(rast[..., -1:] != 0, col, self.bgs), rast, v_ndc, f, pos_gradient_boost=self.boost)
        else:
            v_cols = torch.ones_like(v)
            col = dr.interpolate(v_cols[None, ...], rast, f)[0]
            result = dr.antialias(col, rast, v_ndc, f, pos_gradient_boost=self.boost)
        return result
