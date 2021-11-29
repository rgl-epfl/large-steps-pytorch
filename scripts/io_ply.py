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
        vertex_data = pd.read_csv(filename, sep=" ", header=None, engine="python",
                                     skiprows=top, skipfooter=bottom, usecols=names, names=names)

        for n, col in enumerate(vertex_data.columns):
            vertex_data[col] = vertex_data[col].astype(
                dtypes["vertex"][n][1])

        if mesh_size :
            top = count + points_size

            names = np.array([x[0] for x in dtypes["face"]])
            usecols = [1, 2, 3, 5, 6, 7, 8, 9, 10] if has_texture else [1, 2, 3]
            names = names[usecols]

            data["faces"] = pd.read_csv(
                filename, sep=" ", header=None, engine="python", skiprows=top, usecols=usecols, names=names)

            for n, col in enumerate(data["faces"].columns):
                data["faces"][col] = data["faces"][col].astype(
                    dtypes["face"][n + 1][1])

        # Convert to PyTorch array
        data["vertices"] = torch.tensor(vertex_data[["x", "y", "z"]].values, device='cuda')
        if "nx" in vertex_data.columns:
            data["normals"] = torch.tensor(vertex_data[["nx", "ny", "nz"]].values, device='cuda')
        data["faces"] = torch.tensor(data["faces"].to_numpy(), device='cuda')

    else:
        with open(filename, 'rb') as ply:
            ply.seek(end_header)
            points_np = np.fromfile(ply, dtype=dtypes["vertex"], count=points_size)
            if ext != sys_byteorder:
                points_np = points_np.byteswap().newbyteorder()
            data["vertices"] = torch.tensor(np.stack((points_np["x"],points_np["y"],points_np["z"]), axis=1), device='cuda')
            if "nx" in points_np.dtype.fields.keys():
                data["normals"] = torch.tensor(np.stack((points_np["nx"],points_np["ny"],points_np["nz"]), axis=1), device='cuda')
            if mesh_size:
                mesh_np = np.fromfile(ply, dtype=dtypes["face"], count=mesh_size)
                if ext != sys_byteorder:
                    mesh_np = mesh_np.byteswap().newbyteorder()

                assert (mesh_np["n_points"] == 3*np.ones_like(mesh_np["n_points"])).all(), "Only triangle meshes are supported!"

                data["faces"] = torch.tensor(np.stack((mesh_np["v1"], mesh_np["v2"], mesh_np["v3"]), axis=1), device='cuda')

    return data

def write_ply(filename, v, f, n=None, vc = None, ascii=False):
    """
    Write a mesh as PLY with vertex colors.

    Parameters
    ----------
    filename : str
        Path to which to save the mesh.
    v : numpy.ndarray
        Vertex positions.
    f : numpy.ndarray
        Faces.
    n : numpy.ndarray
        Vertex normals (optional).
    vc : numpy.ndarray
        Vertex colors (optional). Expects colors as floats in [0,1]
    ascii : bool
        Whether we write a text or binary PLY file (defaults to binary as it is more efficient)
    """
    if vc is not None:
        color = (vc*255).astype(np.uint8)
    # headers
    string = 'ply\n'
    if ascii:
        string = string + 'format ascii 1.0\n'
    else:
        string = string + 'format binary_' + sys.byteorder + '_endian 1.0\n'

    string = string + 'element vertex ' + str(v.shape[0]) + '\n'
    string = string + 'property double x\n'
    string = string + 'property double y\n'
    string = string + 'property double z\n'
    if n is not None and n.shape[0] == v.shape[0]:
        string = string + 'property double nx\n'
        string = string + 'property double ny\n'
        string = string + 'property double nz\n'

    if (vc is not None and vc.shape[0] == v.shape[0]):
        string = string + 'property uchar red\n'
        string = string + 'property uchar green\n'
        string = string + 'property uchar blue\n'

    # end of header
    string = string + 'element face ' + str(f.shape[0]) + '\n'
    string = string + 'property list int int vertex_indices\n'
    string = string + 'end_header\n'
    with open(filename, 'w') as file:
        file.write(string)
        if ascii:
            # write vertices
            for ii in range(v.shape[0]):
                string = f"{v[ii,0]} {v[ii,1]} {v[ii,2]}"
                if n is not None and n.shape[0] == v.shape[0]:
                    string = string + f" {n[ii,0]} {n[ii,1]} {n[ii,2]}"
                if (vc is not None and vc.shape[0] == v.shape[0]):
                    string = string + f" {color[ii,0]:03d} {color[ii,1]:03d} {color[ii,2]:03d}\n"
                else:
                    string = string + '\n'
                file.write(string)
            # write faces
            for ii in range(f.shape[0]):
                string = f"3 {f[ii,0]} {f[ii,1]} {f[ii,2]} \n"
                file.write(string)

    if not ascii:
        # Write binary PLY data
        with open(filename, 'ab') as file:
            if vc is None:
                if n is not None:
                    vertex_data = np.hstack((v, n))
                    file.write(vertex_data.astype(np.float64).tobytes())
                else:
                    file.write(v.astype(np.float64).tobytes())
            else:
                if n is not None:
                    vertex_data = np.zeros(v.shape[0], dtype='double,double,double,double,double,double,uint8,uint8,uint8')
                    vertex_data['f0'] = v[:,0].astype(np.float64)
                    vertex_data['f1'] = v[:,1].astype(np.float64)
                    vertex_data['f2'] = v[:,2].astype(np.float64)
                    vertex_data['f3'] = n[:,0].astype(np.float64)
                    vertex_data['f4'] = n[:,1].astype(np.float64)
                    vertex_data['f5'] = n[:,2].astype(np.float64)
                    vertex_data['f6'] = color[:,0]
                    vertex_data['f7'] = color[:,1]
                    vertex_data['f8'] = color[:,2]
                else:
                    vertex_data = np.zeros(v.shape[0], dtype='double,double,double,uint8,uint8,uint8')
                    vertex_data['f0'] = v[:,0].astype(np.float64)
                    vertex_data['f1'] = v[:,1].astype(np.float64)
                    vertex_data['f2'] = v[:,2].astype(np.float64)
                    vertex_data['f3'] = color[:,0]
                    vertex_data['f4'] = color[:,1]
                    vertex_data['f5'] = color[:,2]
                file.write(vertex_data.tobytes())
            # Write faces
            faces = np.hstack((3*np.ones((len(f), 1)), f)).astype(np.int32)
            file.write(faces.tobytes())
