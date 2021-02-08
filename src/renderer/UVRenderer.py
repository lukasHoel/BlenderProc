import bpy

from src.renderer.RendererInterface import RendererInterface
from src.utility.Utility import Utility
from src.utility.RendererUtility import RendererUtility
from src.utility.MaterialLoaderUtility import MaterialLoaderUtility


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
        output = Utility.get_the_one_node_with_type(nodes, 'OutputMaterial')

        links.new(texture_coord_node.outputs['UV'], emission_node.inputs['Color'])
        links.new(emission_node.outputs['Emission'], output.inputs['Surface'])
        return new_mat

    def run(self):
        with Utility.UndoAfterExecution():
            self._configure_renderer()

            new_mat = self._create_uv_material()

            # render normals
            bpy.context.scene.cycles.samples = 1 # this gives the best result for emission shader #TODO test for uv rendering
            bpy.context.view_layer.cycles.use_denoising = False
            for obj in bpy.context.scene.objects:
                if len(obj.material_slots) > 0:
                    for i in range(len(obj.material_slots)):
                        if self._use_alpha_channel:
                            #obj.data.materials[i] = self.add_alpha_texture_node(obj.material_slots[i].material, new_mat)
                            obj.data.materials[i] = MaterialLoaderUtility.add_alpha_texture_node(obj.material_slots[i].material, new_mat)
                        else:
                            obj.data.materials[i] = new_mat
                elif hasattr(obj.data, 'materials'):
                    obj.data.materials.append(new_mat)

            # Set the color channel depth of the output to 32bit
            #bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"
            #bpy.context.scene.render.image_settings.color_depth = "32"
            RendererUtility.set_output_format("OPEN_EXR", 32)

            if self._use_alpha_channel:
                #self.add_alpha_channel_to_textures(blurry_edges=False)
                MaterialLoaderUtility.add_alpha_channel_to_textures(blurry_edges=False)

            #self._render("uvmap_")
            RendererUtility.render(self._determine_output_dir(), "uvmap_", "uvmap")

        Utility.register_output(self._determine_output_dir(), "uvmap_", "uvmap", ".exr", "2.0.0")
        #self._register_output("uvmap_", "uvmap", ".exr", "2.0.0")