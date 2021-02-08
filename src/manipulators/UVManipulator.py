import bpy

from src.main.Module import Module


class UVManipulator(Module):
    def __init__(self, config):
        Module.__init__(self, config)

    def run(self):
        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')

        # looking through all objects
        for obj in bpy.data.objects:
            # if the object is a mesh and not a lamp or camera etc.
            if obj.type == 'MESH':
                obj.select_set(True)
                print("select", obj.name)

        # entering edit mode
        # bpy.ops.object.editmode_toggle()

        # bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(correct_aspect=False, angle_limit=1.2217)

        # exiting edit mode
        # bpy.ops.object.editmode_toggle()
