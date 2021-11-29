import xml.etree.ElementTree as ET
import os
import torch
import numpy as np
from scripts.io_ply import read_ply
import imageio
imageio.plugins.freeimage.download()

def rotation_matrix(axis, angle):
    """
    Builds a homogeneous coordinate rotation matrix along an axis

    Parameters
    ----------

    axis : str
        Axis of rotation, x, y, or z
    angle : float
        Rotation angle, in degrees
    """
    assert axis in 'xyz', "Invalid axis, expected x, y or z"
    mat = torch.eye(4, device='cuda')
    theta = np.deg2rad(angle)
    idx = 'xyz'.find(axis)
    mat[(idx+1)%3, (idx+1)%3] = np.cos(theta)
    mat[(idx+2)%3, (idx+2)%3] = np.cos(theta)
    mat[(idx+1)%3, (idx+2)%3] = -np.sin(theta)
    mat[(idx+2)%3, (idx+1)%3] = np.sin(theta)
    return mat

def translation_matrix(tr):
    """
    Builds a homogeneous coordinate translation matrix

    Parameters
    ----------

    tr : numpy.array
        translation value
    """
    mat = torch.eye(4, device='cuda')
    mat[:3,3] = torch.tensor(tr, device='cuda')
    return mat

def load_scene(filepath):
    """
    Load the meshes, envmap and cameras from a scene XML file.
    We assume the file has the same syntax as Mitsuba 2 scenes.

    Parameters
    ----------

    - filepath : os.path
        Path to the XML file to load
    """
    folder, filename = os.path.split(filepath)
    scene_name, ext = os.path.splitext(filename)
    assert ext == ".xml", f"Unexpected file type: '{ext}'"

    tree = ET.parse(filepath)
    root = tree.getroot()

    assert root.tag == 'scene', f"Unknown root type '{root.tag}', expected 'scene'"

    scene_params = {
        "view_mats" : []
    }

    for plugin in root:
        if plugin.tag == "default":
            if plugin.attrib["name"] == "resx":
                scene_params["res_x"] = int(plugin.attrib["value"])
            elif plugin.attrib["name"] == "resy":
                scene_params["res_y"] = int(plugin.attrib["value"])
        elif plugin.tag == "sensor":
            view_mats = scene_params["view_mats"]
            view_mat = torch.eye(4, device='cuda')
            for prop in plugin:

                if prop.tag == "float":
                    if prop.attrib["name"] == "fov" and "fov" not in scene_params.keys():
                        scene_params["fov"] = float(prop.attrib["value"])
                    elif prop.attrib["name"] == "near_clip" and "near_clip" not in scene_params.keys():
                        scene_params["near_clip"] = float(prop.attrib["value"])
                    elif prop.attrib["name"] == "far_clip" and "far_clip" not in scene_params.keys():
                        scene_params["far_clip"] = float(prop.attrib["value"])
                elif prop.tag == "transform":
                    for tr in prop:
                        if tr.tag == "rotate":
                            if "x" in tr.attrib:
                                view_mat = rotation_matrix("x", float(tr.attrib["angle"])) @ view_mat
                            elif "y" in tr.attrib:
                                view_mat = rotation_matrix("y", float(tr.attrib["angle"])) @ view_mat
                            else:
                                view_mat = rotation_matrix("z", float(tr.attrib["angle"])) @ view_mat
                        elif tr.tag == "translate":
                            view_mat = translation_matrix(np.fromstring(tr.attrib["value"], dtype=float, sep=" ")) @ view_mat
                        else:
                            raise NotImplementedError(f"Unsupported transformation tag: '{tr.tag}'")
            view_mats.append(view_mat.inverse())
        elif plugin.tag == "emitter" and plugin.attrib["type"] == "envmap":
            for prop in plugin:
                if prop.tag == "string" and prop.attrib["name"] == "filename":
                    envmap_path = os.path.join(folder, prop.attrib["value"])
                    envmap = torch.tensor(imageio.imread(envmap_path, format='HDR-FI'), device='cuda')
                    # Add alpha channel
                    alpha = torch.ones((*envmap.shape[:2],1), device='cuda')
                    scene_params["envmap"] = torch.cat((envmap, alpha), dim=-1)
                elif prop.tag == "float" and prop.attrib["name"] == "scale":
                    scene_params["envmap_scale"] = float(prop.attrib["value"])
        elif plugin.tag == "shape":
            if plugin.attrib["type"] == "ply":
                for prop in plugin:
                    if prop.tag == "string" and prop.attrib["name"] == "filename":
                        mesh_path = os.path.join(folder, prop.attrib["value"])
                        assert "id" in plugin.attrib.keys(), "Missing mesh id!"
                        scene_params[plugin.attrib["id"]] = read_ply(mesh_path)
            else:
                raise NotImplementedError(f"Unsupported file type '{plugin.attrib['type']}', only PLY is supported currently")

    assert "mesh-source" in scene_params.keys(), "Missing source mesh"
    assert "mesh-target" in scene_params.keys(), "Missing target mesh"
    assert "envmap" in scene_params.keys(), "Missing envmap"
    assert len(scene_params["view_mats"]) > 0, "At least one camera needed"

    return scene_params
