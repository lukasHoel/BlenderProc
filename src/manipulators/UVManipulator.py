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
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)

        # entering edit mode
        bpy.ops.object.editmode_toggle()

        # select each vertex - only the selected vertices will be considered by uv.smart_project
        bpy.ops.mesh.select_all(action='SELECT')

        # create the uv mapping - this will take a while (i.e. 5-10 minutes for a big scene)
        bpy.ops.uv.smart_project(correct_aspect=False, angle_limit=1.2217)

        # exiting edit mode
        bpy.ops.object.editmode_toggle()
