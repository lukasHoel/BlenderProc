import bpy

from src.main.Module import Module


class UVManipulator(Module):
    def __init__(self, config):
        Module.__init__(self, config)

    def run(self):
        objects = bpy.context.scene.objects

        for obj in objects:
            obj.select_set(True)

        # entering edit mode
        bpy.ops.object.editmode_toggle()

        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(correct_aspect=False, angle_limit=1.2217)

        # exiting edit mode
        bpy.ops.object.editmode_toggle()