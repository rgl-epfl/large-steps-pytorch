# This file is inspired from the pyntcloud project https://github.com/daavoo/pyntcloud/blob/master/pyntcloud/io/ply.py
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import torch

sys_byteorder = ('>', '<')[sys.byteorder == 'little']

ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'b1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}

def read_ply(filename):
    """ Read a .ply (binary or ascii) file and store the elements in pandas DataFrame
    Parameters
    ----------
    filename: str
        Path to the filename
    Returns
    -------
    data: dict
        Elements as pandas DataFrames; comments and ob_info as list of string
    """

    with open(filename, 'rb') as ply:

        if b'ply' not in ply.readline():
            raise ValueError('The file does not start whith the word ply')
        # get binary_little/big or ascii
        fmt = ply.readline().split()[1].decode()
        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        line = []
        dtypes = defaultdict(list)
        count = 2
        points_size = None
        mesh_size = None
        has_texture = False
        comments = []
        while b'end_header' not in line and line != b'':
            line = ply.readline()

            if b'element' in line:
                line = line.split()
                name = line[1].decode()
                size = int(line[2])
                if name == "vertex":
                    points_size = size
                elif name == "face":
                    mesh_size = size

            elif b'property' in line:
                line = line.split()
                # element mesh
                if b'list' in line:

                    if b"vertex_indices" in line[-1] or b"vertex_index" in line[-1]:
                        mesh_names = ["n_points", "v1", "v2", "v3"]
                    else:
                        has_texture = True
                        mesh_names = ["n_coords"] + ["v1_u", "v1_v", "v2_u", "v2_v", "v3_u", "v3_v"]

                    if fmt == "ascii":
                        # the first number has different dtype than the list
                        dtypes[name].append(
                            (mesh_names[0], ply_dtypes[line[2]]))
                        # rest of the numbers have the same dtype
                        dt = ply_dtypes[line[3]]
                    else:
                        # the first number has different dtype than the list
                        dtypes[name].append(
                            (mesh_names[0], ext + ply_dtypes[line[2]]))
                        # rest of the numbers have the same dtype
                        dt = ext + ply_dtypes[line[3]]

                    for j in range(1, len(mesh_names)):
                        dtypes[name].append((mesh_names[j], dt))
                else:
                    if fmt == "ascii":
                        dtypes[name].append(
                            (line[2].decode(), ply_dtypes[line[1]]))
                    else:
                        dtypes[name].append(
                            (line[2].decode(), ext + ply_dtypes[line[1]]))

            elif b'comment' in line:
                line = line.split(b" ", 1)
                comment = line[1].decode().rstrip()
                comments.append(comment)

            count += 1

        # for bin
        end_header = ply.tell()

    data = {}

    if comments:
        data["comments"] = comments

    if fmt == 'ascii':
        top = count
        bottom = 0 if mesh_size is None else mesh_size

        names = [x[0] for x in dtypes["vertex"]]

        data["points"] = pd.read_csv(filename, sep=" ", header=None, engine="python",
                                     skiprows=top, skipfooter=bottom, usecols=names, names=names)

        for n, col in enumerate(data["points"].columns):
            data["points"][col] = data["points"][col].astype(
                dtypes["vertex"][n][1])

        if mesh_size :
            top = count + points_size

            names = np.array([x[0] for x in dtypes["face"]])
            usecols = [1, 2, 3, 5, 6, 7, 8, 9, 10] if has_texture else [1, 2, 3]
            names = names[usecols]

            data["mesh"] = pd.read_csv(
                filename, sep=" ", header=None, engine="python", skiprows=top, usecols=usecols, names=names)

            for n, col in enumerate(data["mesh"].columns):
                data["mesh"][col] = data["mesh"][col].astype(
                    dtypes["face"][n + 1][1])

    else:
        with open(filename, 'rb') as ply:
            ply.seek(end_header)
            points_np = np.fromfile(ply, dtype=dtypes["vertex"], count=points_size)
            if ext != sys_byteorder:
                points_np = points_np.byteswap().newbyteorder()
            data["vertices"] = torch.tensor(np.stack((points_np["x"],points_np["y"],points_np["z"]), axis=1), device='cuda')
            data["normals"] = torch.tensor(np.stack((points_np["nx"],points_np["ny"],points_np["nz"]), axis=1), device='cuda')
            if mesh_size:
                mesh_np = np.fromfile(ply, dtype=dtypes["face"], count=mesh_size)
                if ext != sys_byteorder:
                    mesh_np = mesh_np.byteswap().newbyteorder()

                assert (mesh_np["n_points"] == 3*np.ones_like(mesh_np["n_points"])).all(), "Only triangle meshes are supported!"

                data["faces"] = torch.tensor(np.stack((mesh_np["v1"], mesh_np["v2"], mesh_np["v3"]), axis=1), device='cuda')

    return data

def write_ply(filename, V, F, VC = None):
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
    if VC is not None:
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
