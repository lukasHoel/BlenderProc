import bpy

from src.utility.CameraUtility import CameraUtility
from src.utility.ItemWriter import ItemWriter
from src.writer.WriterInterface import WriterInterface
from src.utility.BlenderUtility import get_all_mesh_objects

import os


class SceneWriter(WriterInterface):
    """ Writes the complete scene as .ply file with uv coordinates.
    """

    def __init__(self, config):
        WriterInterface.__init__(self, config)

    def run(self):
        """ Collect camera and camera object and write them to a file."""

        # save all object names
        path = os.path.join(self._determine_output_dir(False), "objects.txt")
        with open(path, "w") as f:
            for obj in get_all_mesh_objects():
                f.write(f"{obj.name}\n")

        # save mesh
        path = os.path.join(self._determine_output_dir(False), "scene.ply")
        bpy.ops.export_mesh.ply(filepath=path,
                                use_ascii=False,
                                use_selection=False,
                                use_mesh_modifiers=True,
                                use_normals=False,
                                use_uv_coords=True,
                                use_colors=False)
