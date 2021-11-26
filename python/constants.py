import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
#OUTPUT_DIR = os.path.realpath(os.path.join(ROOT_DIR, "output"))
OUTPUT_DIR = "/home/bnicolet/Documents/OT/output_new"
SCENES_DIR = os.path.realpath(os.path.join(ROOT_DIR, "scenes"))
REMESH_DIR = os.path.join(ROOT_DIR, "botsch-kobbelt-remesher-libigl/build")
BLEND_SCENE = os.path.join(SCENES_DIR, "render.blend")
BLENDER_EXEC = "blender2.8"
