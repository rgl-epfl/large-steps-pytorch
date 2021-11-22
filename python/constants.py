import os

OUTPUT_DIR = os.path.realpath("/home/bnicolet/Documents/OT/output_new/")
SCENES_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "../scenes"))
REMESH_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "botsch-kobbelt-remesher-libigl/build")
BLEND_SCENE = os.path.realpath("/home/bnicolet/Documents/OT/diff_geom/render2.blend")
BLENDER_EXEC = "blender2.8"
