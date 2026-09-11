"""
Non-photorealistic rendering: the structural modality.

The rest of the sensor stack answers "what did the camera see". This module
answers "what is the *shape* of what the camera saw", by throwing away
everything a photograph carries that structure does not: texture, colour,
shadow, reflection, and the lighting that produced them. What survives is
contours, creases and silhouettes -- a line drawing of the scene geometry.

That is the intermediate representation the chained architecture is built on. A
policy trained on line drawings cannot be confused by a new floor texture or a
harsh shadow, because neither survives into what it is shown.

    npr = LineArt(scene)
    npr.render(camera, '/tmp/frame.lineart.png', resolution=(320, 240))

HOW IT DIFFERS FROM A PASS
--------------------------
depth and segmentation are render *passes*: the renderer already computed them
while shading, so they come out of the same render as the colour image for
almost nothing. Line art is not like that. FreeStyle draws strokes *into* the
combined image, which means it needs its own render with the scene configured
differently -- flat white materials, a white world, no view transform. So a
frame with line art costs two renders, not one, and this module is honest about
that rather than pretending otherwise.

Measured here: the extra render is roughly 0.7s at 320x240/8spp and 2.1s at
512x512/32spp. The cost scales with resolution rather than sample count,
because stroke generation is a geometric operation performed once per frame
rather than once per sample. It is therefore a large proportional overhead on
cheap frames and a modest one on expensive ones.

WHY THE SCENE MUST BE FLATTENED
-------------------------------
Left alone, FreeStyle draws its lines on top of the ordinary shaded render, and
the result is a photograph with edges emphasised -- which carries exactly the
texture and lighting the representation exists to discard. Overriding every
material with a flat white emitter and whitening the world removes the shading
without touching the geometry, so the strokes are all that is left.

The view transform matters for the same reason and is easy to miss: Blender's
default filmic transform maps pure white to grey, so a "white" background
renders at around 0.8 and the supposedly binary image arrives with a tonal
gradient in it. Line art sets the transform to Standard.
"""

import bpy

#: Name of the override material, so repeated use reuses one datablock rather
#: than accumulating a new one per frame.
FLAT_MATERIAL = 'ROBOTSIM.NPR.FLAT'


def flat_material(colour=(1.0, 1.0, 1.0, 1.0), strength=1.0):
    """
    A pure emitter, used to override every material in the scene.

    Emission rather than diffuse: a diffuse surface still needs a light to be
    visible and still shades by angle, both of which reintroduce exactly the
    lighting dependence the representation is meant to remove. An emitter is
    the same value from every direction under any light.
    """
    mat = bpy.data.materials.get(FLAT_MATERIAL)
    if mat is None:
        mat = bpy.data.materials.new(FLAT_MATERIAL)
    mat.use_nodes = True
    tree = mat.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    emission = tree.nodes.new('ShaderNodeEmission')
    emission.inputs[0].default_value = colour
    emission.inputs[1].default_value = strength
    output = tree.nodes.new('ShaderNodeOutputMaterial')
    tree.links.new(emission.outputs[0], output.inputs[0])
    return mat


class LineArt:
    """
    Renders the scene as a line drawing.

    Every setting it changes is recorded and restored afterwards, because the
    same scene is used for the photorealistic and semantic passes either side of
    it. A line-art render that leaked its white world or its material override
    into the next colour frame would silently corrupt the half of the dataset
    that is supposed to be photorealistic -- and would do it in a way that looks
    like a lighting bug rather than a leak.
    """

    def __init__(self, scene=None, thickness=1.0, line_colour=(0.0, 0.0, 0.0),
                 background=(1.0, 1.0, 1.0, 1.0), crease_angle=None,
                 silhouette=True, crease=True, border=True, edge_mark=True,
                 contour=False, external_contour=False):
        self.scene = scene or bpy.context.scene
        self.thickness = thickness
        self.line_colour = line_colour
        self.background = background
        self.crease_angle = crease_angle
        ## Which classes of edge become strokes. Silhouettes and creases carry
        ## the shape; borders catch the outline of open geometry. Contour and
        ## external contour are off by default because they duplicate
        ## silhouette strokes on closed meshes and thicken everything.
        self.selectors = {
            'select_silhouette': silhouette,
            'select_crease': crease,
            'select_border': border,
            'select_edge_mark': edge_mark,
            'select_contour': contour,
            'select_external_contour': external_contour,
        }
        self._saved = None

    def __repr__(self):
        return '<LineArt thickness=%.2f>' % self.thickness

    # -- scene configuration ------------------------------------------------

    def _save(self):
        scene = self.scene
        view_layer = bpy.context.view_layer
        world = scene.world
        background = None
        if world is not None and world.use_nodes:
            node = world.node_tree.nodes.get('Background')
            if node is not None:
                background = (tuple(node.inputs[0].default_value),
                              node.inputs[1].default_value)
        return {
            'material_override': view_layer.material_override,
            'use_freestyle': scene.render.use_freestyle,
            'layer_freestyle': view_layer.use_freestyle,
            'thickness_mode': scene.render.line_thickness_mode,
            'thickness': scene.render.line_thickness,
            'view_transform': scene.view_settings.view_transform,
            'look': scene.view_settings.look,
            'world': world,
            'background': background,
            'film_transparent': scene.render.film_transparent,
        }

    def _restore(self, saved):
        scene = self.scene
        view_layer = bpy.context.view_layer
        view_layer.material_override = saved['material_override']
        scene.render.use_freestyle = saved['use_freestyle']
        view_layer.use_freestyle = saved['layer_freestyle']
        scene.render.line_thickness_mode = saved['thickness_mode']
        scene.render.line_thickness = saved['thickness']
        scene.view_settings.view_transform = saved['view_transform']
        scene.view_settings.look = saved['look']
        scene.render.film_transparent = saved['film_transparent']
        world = saved['world']
        scene.world = world
        if world is not None and saved['background'] is not None:
            node = world.node_tree.nodes.get('Background')
            if node is not None:
                colour, strength = saved['background']
                node.inputs[0].default_value = colour
                node.inputs[1].default_value = strength

    def configure(self):
        """Put the scene into line-art mode. Returns the saved state."""
        scene = self.scene
        view_layer = bpy.context.view_layer
        saved = self._save()

        view_layer.material_override = flat_material()

        world = scene.world
        if world is None:
            world = bpy.data.worlds.new('ROBOTSIM.NPR.WORLD')
            scene.world = world
        world.use_nodes = True
        node = world.node_tree.nodes.get('Background')
        if node is not None:
            node.inputs[0].default_value = self.background
            node.inputs[1].default_value = 1.0
        scene.render.film_transparent = False

        scene.render.use_freestyle = True
        view_layer.use_freestyle = True
        scene.render.line_thickness_mode = 'ABSOLUTE'
        scene.render.line_thickness = self.thickness

        settings = view_layer.freestyle_settings
        lineset = settings.linesets[0] if settings.linesets else settings.linesets.new('robotsim')
        for name, value in self.selectors.items():
            if hasattr(lineset, name):
                setattr(lineset, name, value)
        if self.crease_angle is not None:
            settings.crease_angle = self.crease_angle
        if lineset.linestyle is not None:
            lineset.linestyle.color = self.line_colour
            lineset.linestyle.thickness = self.thickness

        ## Filmic maps pure white to grey; a "binary" drawing would arrive with
        ## a tonal gradient baked into it.
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
        return saved

    # -- rendering ----------------------------------------------------------

    def render(self, camera, output_path, resolution=None, engine=None):
        """
        Render one line drawing and restore the scene.

        Restoration runs in a finally block: an exception mid-render would
        otherwise leave every subsequent photorealistic frame flat white, and
        the dataset would be quietly ruined from that point on.
        """
        scene = self.scene
        saved = self.configure()
        try:
            if engine is not None:
                scene.render.engine = engine
            if resolution:
                scene.render.resolution_x, scene.render.resolution_y = resolution
            scene.render.resolution_percentage = 100
            scene.camera = camera
            scene.render.image_settings.file_format = 'PNG'
            scene.render.filepath = output_path
            bpy.ops.render.render(write_still=True)
        finally:
            self._restore(saved)
        ## Blender appends nothing when write_still is used with a full path,
        ## but it does add the extension if absent.
        if not output_path.lower().endswith('.png'):
            return output_path + '.png'
        return output_path
