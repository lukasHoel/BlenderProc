import bpy

from src.utility.CameraUtility import CameraUtility
from src.utility.ItemWriter import ItemWriter
from src.writer.WriterInterface import WriterInterface

import os


class SceneWriter(WriterInterface):
    """ Writes the complete scene as .ply file with uv coordinates.
    """

    def __init__(self, config):
        WriterInterface.__init__(self, config)

    def run(self):
        """ Collect camera and camera object and write them to a file."""

        path = os.path.join(self._determine_output_dir(False), "scene.ply")

        bpy.ops.export_mesh.ply(filepath=path,
                                use_ascii=False,
                                use_selection=False,
                                use_mesh_modifiers=True,
                                use_normals=False,
                                use_uv_coords=True,
                                use_colors=False)
