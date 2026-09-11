"""
Domain randomisation: making the corpus varied, and making it reproducible.

A perception network trained on one room learns that room. The defence is to
vary everything that should not matter -- lighting, texture, colour, layout,
viewpoint -- while holding fixed the structure that should. This module varies
the first list.

Every choice is drawn from a seeded generator owned by this module rather than
from the global `random` state. That is the difference between a dataset and a
snapshot: given the seed, the scene regenerates exactly, so a training run is
reproducible, a suspicious sample can be recovered from its manifest entry
alone, and two experiments differing in one parameter really differ in one
parameter.

    r = Randomizer(seed=1234)
    scene = r.scene(obstacles=(4, 9))
    r.lighting()
    r.camera(cam, looking_at=(0, 0, 0.5))

WHAT IS DELIBERATELY NOT RANDOMISED
-----------------------------------
Scale and gravity direction. Randomising those teaches the network that size and
up are negotiable, which is the opposite of the structural prior the chained
architecture depends on: the policy downstream reasons about geometry, and
geometry it cannot trust is worse than no geometry.
"""

import math
import random

import bpy

#: Shapes used to populate a scene. Deliberately primitive: the point of the
#: structural representation is that a line drawing of a cylinder and a line
#: drawing of a chair present the same problem to the policy -- an obstacle
#: boundary -- so elaborate assets add render cost without adding signal.
SHAPES = ('cube', 'cylinder', 'wall')


class Randomizer:
    """
    Seeded scene randomisation.

    Holds its own `random.Random`, so nothing it does depends on or disturbs
    global random state -- which matters because the simulator, the physics and
    any training code may all be drawing from that global state for their own
    reasons.
    """

    def __init__(self, seed=0, extent=12.0, keep=()):
        self.seed = seed
        self.rng = random.Random(seed)
        ## Half-width of the region scenes are generated in.
        self.extent = extent
        ## Objects that must survive a reset: the robot, the ground, anything
        ## the caller built on purpose.
        self.keep = set(keep)
        self.created = []

    def __repr__(self):
        return '<Randomizer seed=%d created=%d>' % (self.seed, len(self.created))

    def reseed(self, seed):
        """Restart the sequence. The same seed gives the same scene."""
        self.seed = seed
        self.rng = random.Random(seed)
        return self

    # -- scene contents -----------------------------------------------------

    def clear(self):
        """
        Remove everything this randomizer made, leaving the rest alone.

        The object's data is freed too, not just the object. Removing an object
        orphans its mesh or lamp datablock rather than deleting it, and over a
        corpus of thousands of scenes those orphans accumulate into real memory.
        They also take the names with them: the next cube becomes `RND.CUBE.001`
        because the old one still holds `RND.CUBE`, so identical scenes end up
        with differently-named objects, which perturbs render ordering and makes
        a corpus depend on how many samples preceded it.
        """
        for obj in self.created:
            try:
                data = getattr(obj, 'data', None)
                kind = getattr(obj, 'type', None)
                bpy.data.objects.remove(obj, do_unlink=True)
                if data is not None and getattr(data, 'users', 0) == 0:
                    if kind == 'MESH':
                        bpy.data.meshes.remove(data)
                    elif kind == 'LIGHT':
                        bpy.data.lights.remove(data)
            except (ReferenceError, RuntimeError):
                pass
        self.created = []
        return self

    def scene(self, obstacles=(3, 8), clear=True, label=None, min_gap=1.2,
              origin_clearance=2.5):
        """
        Populate the scene with obstacles and return them.

        `origin_clearance` keeps a disc around the origin empty, because a robot
        that spawns inside an obstacle produces a frame whose geometry is
        physically impossible -- and those frames teach the perception network
        that objects interpenetrate.
        """
        if clear:
            self.clear()
        count = self.rng.randint(*obstacles)
        placed = []
        for _ in range(count):
            spot = self._free_spot(placed, min_gap, origin_clearance)
            if spot is None:
                ## The scene is full. Stopping early is correct: forcing the
                ## requested count would mean overlapping geometry.
                break
            obj = self._make_shape(spot)
            if label is not None:
                obj.pass_index = label
            placed.append((spot, obj))
            self.created.append(obj)
        return [obj for _spot, obj in placed]

    def _free_spot(self, placed, min_gap, origin_clearance, attempts=40):
        for _ in range(attempts):
            x = self.rng.uniform(-self.extent, self.extent)
            y = self.rng.uniform(-self.extent, self.extent)
            if math.hypot(x, y) < origin_clearance:
                continue
            if all(math.hypot(x - sx, y - sy) >= min_gap
                   for (sx, sy), _obj in placed):
                return (x, y)
        return None

    def _make_shape(self, spot):
        x, y = spot
        kind = self.rng.choice(SHAPES)
        if kind == 'cylinder':
            radius = self.rng.uniform(0.25, 0.8)
            height = self.rng.uniform(1.0, 3.0)
            obj = _create('cylinder', radius=radius, depth=height,
                          location=(x, y, height * 0.5))
            ## Cylinders are authored lying along X; stand it up.
            obj.rotation_euler.y = math.pi / 2
        elif kind == 'wall':
            length = self.rng.uniform(2.0, 6.0)
            height = self.rng.uniform(0.8, 2.5)
            obj = _create('cube', size=(length, self.rng.uniform(0.2, 0.5), height),
                          location=(x, y, height * 0.5))
            obj.rotation_euler.z = self.rng.uniform(0, math.pi)
        else:
            side = self.rng.uniform(0.4, 1.6)
            height = self.rng.uniform(0.4, 2.0)
            obj = _create('cube', size=(side, side, height),
                          location=(x, y, height * 0.5))
            obj.rotation_euler.z = self.rng.uniform(0, math.pi)
        return obj

    # -- appearance ---------------------------------------------------------

    def materials(self, objects, saturation=(0.0, 0.9), value=(0.05, 0.95)):
        """
        Give each object a random flat colour.

        Appearance only. The semantic pass reads `pass_index`, not colour, so
        randomising materials cannot disturb the labels -- which is the whole
        reason the ObjectID bridge uses an index rather than a rendered colour.
        """
        for obj in objects:
            if getattr(obj, 'type', None) != 'MESH':
                continue
            mat = bpy.data.materials.new('RND.%d' % self.rng.randrange(1 << 30))
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get('Principled BSDF')
            if bsdf is not None:
                import colorsys
                h = self.rng.random()
                s = self.rng.uniform(*saturation)
                v = self.rng.uniform(*value)
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)
                if 'Roughness' in bsdf.inputs:
                    bsdf.inputs['Roughness'].default_value = self.rng.uniform(0.05, 1.0)
                if 'Metallic' in bsdf.inputs:
                    ## Occasionally metallic, to produce the specular highlights
                    ## that defeat photorealistic models -- the adversarial case
                    ## the structural representation is supposed to survive.
                    bsdf.inputs['Metallic'].default_value = (
                        self.rng.uniform(0.6, 1.0) if self.rng.random() < 0.25 else 0.0)
            obj.data.materials.clear()
            obj.data.materials.append(mat)
        return objects

    def lighting(self, count=(1, 3), energy=(2.0, 12.0), warmth=(3000, 9000)):
        """Random sun angles, strengths and colour temperatures."""
        lights = []
        for _ in range(self.rng.randint(*count)):
            data = bpy.data.lights.new('RND.SUN', type='SUN')
            data.energy = self.rng.uniform(*energy)
            data.angle = self.rng.uniform(0.0, 0.4)
            kelvin = self.rng.uniform(*warmth)
            data.color = _kelvin_to_rgb(kelvin)
            obj = bpy.data.objects.new('RND.SUN', data)
            bpy.context.scene.collection.objects.link(obj)
            ## Elevation kept above the horizon: a sun below it lights nothing
            ## and wastes the sample.
            obj.rotation_euler = (self.rng.uniform(0.15, 1.2),
                                  0.0,
                                  self.rng.uniform(0, 2 * math.pi))
            self.created.append(obj)
            lights.append(obj)
        return lights

    def world(self, brightness=(0.02, 0.6)):
        """Random ambient level, standing in for an environment map."""
        scene = bpy.context.scene
        world = scene.world
        if world is None:
            world = bpy.data.worlds.new('RND.WORLD')
            scene.world = world
        world.use_nodes = True
        node = world.node_tree.nodes.get('Background')
        if node is not None:
            level = self.rng.uniform(*brightness)
            tint = self.rng.uniform(0.85, 1.0)
            node.inputs[0].default_value = (level * tint, level * tint, level, 1.0)
            node.inputs[1].default_value = 1.0
        return world

    # -- viewpoint ----------------------------------------------------------

    def camera(self, camera, looking_at=(0.0, 0.0, 0.5), distance=(4.0, 12.0),
               elevation=(0.15, 0.9), lens=(24.0, 60.0)):
        """
        Place a camera on a random arc around a point, aimed at it.

        Aimed rather than randomly oriented: a viewpoint that misses the scene
        produces an empty frame, which costs a full render and teaches nothing.
        """
        import mathutils
        target = mathutils.Vector(looking_at)
        azimuth = self.rng.uniform(0, 2 * math.pi)
        radius = self.rng.uniform(*distance)
        pitch = self.rng.uniform(*elevation)
        offset = mathutils.Vector((math.cos(azimuth) * math.cos(pitch),
                                   math.sin(azimuth) * math.cos(pitch),
                                   math.sin(pitch))) * radius
        camera.location = target + offset
        direction = (target - camera.location).normalized()
        camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        if hasattr(camera.data, 'lens'):
            camera.data.lens = self.rng.uniform(*lens)
        return camera

    def pose(self, obj, area=None, yaw=(0.0, 2 * math.pi), z=None):
        """Random position and heading for an object, e.g. the robot."""
        span = self.extent if area is None else area
        obj.location.x = self.rng.uniform(-span, span)
        obj.location.y = self.rng.uniform(-span, span)
        if z is not None:
            obj.location.z = z
        obj.rotation_euler.z = self.rng.uniform(*yaw)
        return obj


def _create(kind, **kw):
    """Build a primitive without importing robotsim (which would need bpy)."""
    import robotsim
    if kind == 'cylinder':
        return robotsim.create_cylinder('RND.CYL', **kw)
    return robotsim.create_cube('RND.CUBE', **kw)


def _kelvin_to_rgb(kelvin):
    """
    Approximate blackbody colour, normalised to the brightest channel.

    Cheap analytic fit rather than Blender's blackbody node: this only needs to
    make lighting vary plausibly across a corpus, and a node would have to be
    wired into every light's node tree for no additional realism at this scale.
    """
    t = max(1000.0, min(12000.0, kelvin)) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        r = 329.698727446 * ((t - 60) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60) ** -0.0755148492)
    if t >= 66:
        b = 255.0
    elif t <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10) - 305.0447927307
    channels = [max(0.0, min(255.0, c)) / 255.0 for c in (r, g, b)]
    peak = max(channels) or 1.0
    return tuple(c / peak for c in channels)
