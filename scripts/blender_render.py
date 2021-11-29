import bpy
import os

import argparse
import sys
import numpy as np

# Remove annoying argument that messes up argparse
sys.argv.remove('--')

parser = argparse.ArgumentParser(description="Render OBJ meshes from a given viewpoint, with wireframe.")
parser.add_argument("--input", "-i", required=True, type=os.path.abspath, help="Meshes to render.", nargs="+")
parser.add_argument("--collection", "-c", type=str, default="14", help="Camera collection to use.")
parser.add_argument("--smooth", "-s", action="store_true", help="Render without wireframe.")
parser.add_argument("--thickness", "-t", type=float, default=0.008, help="Thickness of the wireframe.")
parser.add_argument("--viewpoint", "-v", type=int, default=0, help="Index of the camera with which to render.")
parser.add_argument("--output", "-o", type=os.path.abspath, help="Output directory.")
parser.add_argument("--resolution", "-r", type=float, default=100, help="Rendering resolution fraction.")
parser.add_argument("--background", action="store_true", help="Render the background or not.")
parser.add_argument("--ours", action="store_true", help="Color mesh as ours")
parser.add_argument("--baseline", action="store_true", help="Color mesh as baseline")
parser.add_argument("--area", action="store_true", help="Color mesh depending on surface area")
parser.add_argument("--sequence", action="store_true", help="Handle naming so that it is compatible with premiere sequence import")
parser.add_argument("--lines", action="store_true", help="Show self intersection as lines")
parser.add_argument("--faces", action="store_true", help="Show self intersection as faces")
parser.add_argument("--it", type=int, default=-1)
# Parse command line args
params = parser.parse_known_args()[0]
if params.sequence and params.it == -1:
    raise ValueError("Invalid iteration number!")
if params.output is None:
    params.output = os.getcwd()
if not os.path.isdir(params.output):
    os.makedirs(params.output)

assert params.collection in bpy.data.collections.keys(), "Wrong collection name!"

bpy.ops.object.select_all(action='DESELECT')
white_mat = bpy.data.materials["White"]
baseline_mat = bpy.data.materials["Baseline"]
ours_mat = bpy.data.materials["Ours"]
black_mat = bpy.data.materials["Black"]
area_mat = bpy.data.materials["Area"]
lines_mat = bpy.data.materials["Intersections"]

i=0
for filename in params.input:
    folder, obj_file = os.path.split(filename)
    name, ext = os.path.splitext(obj_file)
    # Import the object
    if ext == ".obj":
        bpy.ops.import_scene.obj(filepath=filename)
    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=filename)
    else:
        raise ValueError(f"Unsupported extension: {ext} ! This script only supports OBJ and PLY files.")
    # Make the imported object the active one
    obj = bpy.context.selected_objects[-1]
    bpy.context.view_layer.objects.active = obj
    if ext == ".ply":
        obj.rotation_euler[0] = np.pi / 2
    # Assign materials
    assert len(obj.data.materials) == (1 if ext==".obj" else 0)
    bpy.ops.object.material_slot_add()
    if ext == ".ply":
        bpy.ops.object.material_slot_add()

    if params.area:
        obj.data.materials[0] = area_mat
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 3
    elif params.ours:
        obj.data.materials[0] = ours_mat
    elif params.baseline:
        obj.data.materials[0] = baseline_mat
    else:
        obj.data.materials[0] = white_mat
    obj.data.materials[1] = black_mat
    if not params.smooth:
        # Add Wireframe
        bpy.ops.object.modifier_add(type='WIREFRAME')
        obj.modifiers["Wireframe"].use_replace = False
        obj.modifiers["Wireframe"].use_even_offset = False
        obj.modifiers["Wireframe"].material_offset = 1
        obj.modifiers["Wireframe"].thickness = params.thickness

    # If needed, load the lines
    if params.lines:
        bpy.ops.import_scene.obj(filepath=os.path.join(folder, f"{name}_lines.obj"))
        # Make the imported object the active one
        lines = bpy.context.selected_objects[-1]
        bpy.context.view_layer.objects.active = lines
        # Convert to a curve
        bpy.ops.object.convert(target='CURVE')
        # Set bevel
        lines.data.bevel_depth = 0.008
        # Assign material
        assert len(lines.data.materials) == 0
        bpy.ops.object.material_slot_add()
        lines.data.materials[0] = lines_mat
    elif params.faces:
        # Load conflicting faces and assign material
        mask = np.genfromtxt(os.path.join(folder, f"{name}.csv"), delimiter=" ", dtype=int)
        bpy.ops.object.material_slot_add()
        obj.data.materials[2] = lines_mat
        for i in mask:
            obj.data.polygons[i].material_index = 2

    # Set the active camera
    bpy.data.scenes["Scene"].camera = bpy.data.collections[params.collection].objects[params.viewpoint]
    # Render
    bpy.data.scenes["Scene"].render.film_transparent = not params.background
    bpy.data.scenes["Scene"].render.resolution_percentage = params.resolution
    if params.sequence:
        bpy.data.scenes["Scene"].render.filepath = os.path.join(params.output, f"{'smooth' if params.smooth else 'wireframe'}_{params.it:04d}.png")
    else:
        bpy.data.scenes["Scene"].render.filepath = os.path.join(params.output, f"{name}_{'smooth' if params.smooth else 'wireframe'}.png")
    bpy.ops.render.render(write_still=True)

    # Delete the object
    bpy.ops.object.delete()
    i+=1
