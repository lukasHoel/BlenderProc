import bpy

from src.main.Module import Module
from src.utility.BlenderUtility import get_all_mesh_objects


class SmartUVManipulator(Module):
    """
    Uses smart uv project ot re-parameterize all objects in the scene

    Example:

    .. code-block:: yaml

        {
          "module": "manipulators.SmartUVManipulator",
          "config": {
            "angle_limit": 1.2217
          }
        }
    """

    def __init__(self, config):
        Module.__init__(self, config)

    def run(self):
        """ Assigns user-defined values to World's attributes, custom properties, or manipulates the state of the world.
            1. Selects current active World.
            2. Change World's state via setting user-defined values to it's attributes, custom properties, etc.
        """

        angle_limit = self.config.get_float("angle_limit", 1.2217)

        # entering edit mode
        bpy.ops.object.editmode_toggle()
        # select all objects elements
        bpy.ops.mesh.select_all(action='SELECT')
        # the actual unwrapping operation, 1.2217 are 70 degrees
        bpy.ops.uv.smart_project(correct_aspect=False, angle_limit=angle_limit)
        # exiting edit mode
        bpy.ops.object.editmode_toggle()
