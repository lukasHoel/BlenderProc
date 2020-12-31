import bpy

from src.renderer.RendererInterface import RendererInterface
from src.utility.Utility import Utility
from src.utility.BlenderUtility import get_all_mesh_objects

import os

class UVRenderer(RendererInterface):
    """  Renders uv images for each registered key point.

    The rendering is stored using the .exr file type and a color depth of 32bit
    to achieve high precision. Furthermore, this renderer does not support anti-aliasing.
    """

    def __init__(self, config):
        RendererInterface.__init__(self, config)

    def _create_uv_material(self):
        """ Creates a new material which uses (u,v,0) texture coordinates as rgb.

        This assumes a linear color space used for rendering!
        """
        new_mat = bpy.data.materials.new(name="Normal")
        new_mat.use_nodes = True
        nodes = new_mat.node_tree.nodes

        # clean material from Principled BSDF nodes
        for node in Utility.get_nodes_with_type(nodes, "BsdfPrincipled"):
            nodes.remove(node)

        links = new_mat.node_tree.links
        texture_coord_node = nodes.new(type='ShaderNodeTexCoord')
        emission_node = nodes.new(type='ShaderNodeEmission')
        emission_node.inputs['Strength'].default_value = 1
        # TODO set intensity of emission_node to 1.0 ( see emission docu online)
        output = Utility.get_the_one_node_with_type(nodes, 'OutputMaterial')

        vector_transform_node = nodes.new(type='ShaderNodeVectorTransform')
        vector_transform_node.vector_type = "VECTOR"
        vector_transform_node.convert_from = "OBJECT"
        vector_transform_node.convert_to = "CAMERA"

        #links.new(texture_coord_node.outputs['UV'], vector_transform_node.inputs['Vector'])
        #links.new(vector_transform_node.outputs['Vector'], output.inputs['Surface'])

        links.new(texture_coord_node.outputs['UV'], emission_node.inputs['Color'])
        # TODO pass this first to a node that converts it to [0, 1] range? why is it not in this range?
        links.new(emission_node.outputs['Emission'], output.inputs['Surface'])

        # TODO think about logging the outputs of the nodes? https://devtalk.blender.org/t/is-it-possible-to-read-the-output-values-of-nodes-from-a-material/7899


        #links.new(texture_coord_node.outputs['UV'], output.inputs['Surface'])
        return new_mat

    def run(self):
        with Utility.UndoAfterExecution():
            self._configure_renderer()


            for ob in get_all_mesh_objects():
                # Loops per face
                for face in ob.data.polygons:
                    for vert_idx, loop_idx in zip(face.vertices, face.loop_indices):
                        uv_coords = ob.data.uv_layers.active.data[loop_idx].uv
                        if uv_coords.x < 0 or uv_coords.x > 1 or uv_coords.y < 0 or uv_coords.y > 1:
                            print("face idx: %i, vert idx: %i, uvs: %f, %f" % (face.index, vert_idx, uv_coords.x, uv_coords.y))
            raise ValueError("debug")


            new_mat = self._create_uv_material()

            # render normals
            bpy.context.scene.cycles.samples = 1 # this gives the best result for emission shader #TODO test for uv rendering
            bpy.context.view_layer.cycles.use_denoising = False
            for obj in bpy.context.scene.objects:
                if len(obj.material_slots) > 0:
                    for i in range(len(obj.material_slots)):
                        if self._use_alpha_channel:
                            obj.data.materials[i] = self.add_alpha_texture_node(obj.material_slots[i].material, new_mat)
                        else:
                            obj.data.materials[i] = new_mat
                elif hasattr(obj.data, 'materials'):
                    obj.data.materials.append(new_mat)

            # Set the color channel depth of the output to 32bit
            bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"
            bpy.context.scene.render.image_settings.color_depth = "32"

            if self._use_alpha_channel:
                self.add_alpha_channel_to_textures(blurry_edges=False)

            self._render("uvmap_")

        self._register_output("uvmap_", "uvmap", ".exr", "2.0.0")

    def _render_opengl(self, default_prefix, custom_file_path=None):
        """ Renders each registered keypoint.

        :param default_prefix: The default prefix of the output files.
        """
        if custom_file_path is None:
            bpy.context.scene.render.filepath = os.path.join(self._determine_output_dir(),
                                                             self.config.get_string("output_file_prefix",
                                                                                    default_prefix))
        else:
            bpy.context.scene.render.filepath = custom_file_path

        # Skip if there is nothing to render
        if bpy.context.scene.frame_end != bpy.context.scene.frame_start:
            if len(get_all_mesh_objects()) == 0:
                raise Exception("There are no mesh-objects to render, "
                                "please load an object before invoking the renderer.")
            # As frame_end is pointing to the next free frame, decrease it by one, as
            # blender will render all frames in [frame_start, frame_ned]
            bpy.context.scene.frame_end -= 1
            if not self._avoid_rendering:
                bpy.ops.render.opengl(animation=True, write_still=True)
            # Revert changes
            bpy.context.scene.frame_end += 1