import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.realpath(os.path.join(ROOT_DIR, "output")) # Change this if you want to output the data and figures elsewhere
SCENES_DIR = os.path.realpath(os.path.join(ROOT_DIR, "scenes"))
REMESH_DIR = os.path.join(ROOT_DIR, "ext/botsch-kobbelt-remesher-libigl/build")
BLEND_SCENE = os.path.join(SCENES_DIR, "render.blend")
BLENDER_EXEC = "blender2.8" # Change this if you have a different blender installation
