#!/bin/sh
"exec" "blender" "--python-exit-code" "1" "--python" "$0" "--" "$@"
import os, sys, bpy, math
print(bpy)
HERE = os.path.split(__file__)[0]
# robotsim.py is run as __main__ by Blender, so its own directory is not
# automatically importable. Add it so sibling modules (kinematics, etc) resolve
# whether we are launched directly or via headless.py.
if HERE not in sys.path:
    sys.path.append(HERE)
## ros2/ holds the export layer; adding it keeps `import joint_export` working
## from test scripts exec'd into these globals.
_ROS2 = os.path.join(HERE, 'ros2')
if _ROS2 not in sys.path:
    sys.path.append(_ROS2)
from kinematics import Arm, Joint
from sensors import Lidar, LidarScan, NO_RETURN, beam_angles, ray_direction
from contact import RayContact, Contact
import npr
from npr import LineArt
import randomize
from randomize import Randomizer
import dataset
from dataset import Dataset
import telemetry
from telemetry import Telemetry
import firmware
from firmware import Board, FirmwareDrive, Network, point_to_point, connect
from drive import (DriveBase, DifferentialDrive, AckermannDrive, DRIVE_MODELS,
                   ContactModel, Wheel, wheel_layout, layout_track, layout_wheelbase)
from recorder import Recorder
# Extract arguments safely
argv = sys.argv
print(argv)
if "--" in argv: script_args = argv[argv.index("--") + 1:]
else: script_args = []
print("script arguments:", script_args)
if not script_args: script_args = []  ## load a default robot arm

if 'Cube' in bpy.data.objects:
    bpy.data.objects['Cube'].scale = [100,100,1]
    bpy.data.objects['Cube'].location.z = -1

DECIMATE_THRESHOLD = 3000   ## leave meshes at or under this alone
DECIMATE_TARGET = 1000      ## collapse anything bigger down to roughly this


def weld_mesh(obj, distance=1e-5):
    """
    Merge duplicate vertices in place, returning (before, after).

    CAD exports routinely arrive unwelded -- ur10's link_4 has 17302 vertices
    but only 13443 triangles, because vertices are split per face rather than
    shared. Collapse decimation cannot merge across those splits, so it hits a
    floor far above the requested target (that mesh bottomed out at ~6700).
    Welding first both removes the duplicates outright and lets decimation
    actually reach the target.

    Uses bmesh rather than bpy.ops so it needs no mode switching or active
    object, which makes it safe to call during a load.
    """
    import bmesh
    me = obj.data
    before = len(me.vertices)
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=distance)
    bm.to_mesh(me)
    bm.free()
    me.update()
    return (before, len(me.vertices))


def decimate_mesh(obj, threshold=DECIMATE_THRESHOLD, target=DECIMATE_TARGET, apply=True,
                  max_passes=3, tolerance=1.25, weld=True):
    """
    Collapse a heavy mesh down to roughly `target` vertices.

    The stock arm rigs are CAD exports -- irb120 is ~50k vertices and ur10 ~88k,
    almost all of it detail that survives no visible distance in a 64x32 camera
    render. Every one of those vertices is paid for on each depsgraph evaluation
    and each render, i.e. many times per simulated second.

    Duplicate vertices are welded first; see weld_mesh() for why that is what
    makes hitting the target possible at all.

    The modifier's ratio is a ratio of *faces*, not vertices, so a ratio computed
    from the vertex count consistently under-decimates. Since the true count is
    only knowable after applying, this measures the result and takes another
    pass if it is still well over target.

    The modifier is applied rather than left live where possible, so the cost is
    paid once at load instead of on every evaluation. Applying fails on linked
    library data and on multi-user meshes; in that case the modifier is left in
    place, which still cuts render cost, just not evaluation cost.

    Returns (before, after) vertex counts, or None if the mesh was left alone.
    """
    if obj is None or obj.type != 'MESH' or obj.data is None:
        return None
    before = len(obj.data.vertices)
    if before <= threshold:
        return None

    if weld:
        weld_mesh(obj)

    for _ in range(max(1, max_passes)):
        current = len(obj.data.vertices)
        if current <= target * tolerance:
            break
        mod = obj.modifiers.new('ROBOTSIM.DECIMATE', 'DECIMATE')
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = max(min(target / float(current), 1.0), 0.0)
        try:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except (RuntimeError, ReferenceError) as e:
            ## Linked or multi-user data: keep the modifier live instead. It
            ## cannot be measured or iterated on, so stop here.
            print('decimate: could not apply on %s (%s)' % (obj.name, e))
            break
        if not apply:
            break
    return (before, len(obj.data.vertices))


def load_blend_objects(path, link=False, skip=['Camera', 'Plane', 'Camera.001'],
                       decimate=True, decimate_threshold=DECIMATE_THRESHOLD,
                       decimate_target=DECIMATE_TARGET):
    """
    Links or appends all objects from a .blend file into the active scene.
    Returns a dictionary mapping the file path to a list of loaded object references.

    Heavy meshes are decimated on load by default; pass decimate=False to keep
    the source geometry (for a final high-quality render, say).
    """            
    file_objects = []
    # Open the external library and target its objects
    with bpy.data.libraries.load(path, link=link) as (data_from, data_to):
        # Assigning the list tells Blender to load all these objects.
        # Blender automatically resolves and imports dependencies (armatures, materials).
        data_to.objects = data_from.objects
    
    # The objects are now in bpy.data, but not yet visible in the scene.
    # We must link them to the active scene collection.
    saved = 0
    for obj in data_to.objects:
        if obj is not None:
            s = False
            for n in skip:
                if obj.name.startswith(n):
                    print('skip:', obj.name)
                    s = True
                    break
            if s: continue
            print('loading:', obj.name)
            if obj.name not in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.link(obj)
            file_objects.append(obj)
            if obj.type=='ARMATURE':
                print('ARMATURE:', obj.name)
                obj.pose.ik_solver = "LEGACY"  ## force standard IK solver
            elif decimate and obj.type == 'MESH':
                ## Must happen after the object is linked into the scene, or
                ## modifier_apply has no context to operate in.
                result = decimate_mesh(obj, decimate_threshold, decimate_target)
                if result:
                    before, after = result
                    saved += before - after
                    print('decimate: %s %d -> %d' % (obj.name, before, after))
    if saved:
        print('decimate: dropped %d vertices from %s' % (saved, os.path.basename(path)))
    return file_objects


def create_empty(name="Empty"):
    """
    Creates a new Empty object, links it to the active scene collection, 
    and returns the object reference.
    """
    # Create a new empty data-block
    empty_data = bpy.data.objects.new(name, None)
    
    # Set the empty display type if desired (e.g., 'PLAIN_AXES', 'CUBE', 'SPHERE')
    empty_data.empty_display_type = 'PLAIN_AXES'
    
    # Link the empty object to the current scene's collection
    bpy.context.scene.collection.objects.link(empty_data)
    
    return empty_data

def create_cube(name="Cube", size=(1,1,1), location=(0.0, 0.0, 0.0)):
    """
    Creates a new mesh cube with individual X, Y, Z dimensions, 
    links it to the active scene collection, and returns the object reference.
    """
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    cube_obj = bpy.context.active_object
    cube_obj.name = name
    cube_obj.scale = size    
    # Apply the scale so the object's base scale resets to (1, 1, 1).
    # location and rotation must be passed explicitly: the operator's RNA
    # defaults are all True, so transform_apply(scale=True) *also* applies the
    # location -- baking the offset into the mesh and leaving the object sitting
    # at the origin. That still renders correctly, so it went unnoticed, but it
    # means the object's transform is a lie and nothing can be parented to it.
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return cube_obj

def create_cylinder(name="Cylinder", radius=0.1, depth=1, location=(0,0,0)):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth)
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler.y = math.pi / 2
    ## Explicit for the same reason as create_cube: the operator's defaults are
    ## all True. Harmless here only because location is still the origin at this
    ## point, but relying on that is a trap.
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    obj.location = location
    return obj

def create_camera(name="camera", location=(0,0,0)):
    bpy.ops.object.camera_add()
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler.x = math.pi / 2
    if name == 'left': obj.rotation_euler.z = math.pi / 2
    elif name == 'right': obj.rotation_euler.z = -math.pi / 2
    elif name == 'back': obj.rotation_euler.z = math.pi
    obj.location = location
    obj.data.display_size = 0.2
    obj.data.lens = 16
    return obj


def set_if_exists(owner, attr, value):
    """
    Set an RNA property only if this Blender version actually has it.
    Render-engine settings get added and removed between releases, so probing
    keeps quick_render() working across 3.x / 4.x instead of raising AttributeError.
    """
    if hasattr(owner, attr):
        try:
            setattr(owner, attr, value)
            return True
        except (AttributeError, TypeError) as e:
            print('set_if_exists failed:', attr, e)
    return False


RENDER_ENGINES = {
    'workbench': 'BLENDER_WORKBENCH',
    'eevee': 'BLENDER_EEVEE',
    'cycles': 'CYCLES',
}


def set_render_engine(engine):
    """
    Switch the scene's render engine by short name ('workbench', 'eevee',
    'cycles') or by Blender's own identifier.

    Worth knowing before choosing: in background mode EEVEE costs a fixed ~2.2s
    per render on this machine *regardless of resolution* -- it is per-render
    context setup, not pixels -- while Workbench costs ~0.08s. Dropping the
    resolution therefore buys almost nothing on EEVEE; changing engine, cutting
    the number of cameras, or capturing less often are the real levers.

    Workbench draws solid shaded geometry with no lights, shadows or materials,
    which is wrong for training data but ideal for tests that only need to prove
    a camera pointed somewhere and produced pixels.
    """
    name = RENDER_ENGINES.get(engine, engine)
    scene = bpy.context.scene
    try:
        scene.render.engine = name
    except TypeError:
        ## EEVEE was renamed BLENDER_EEVEE_NEXT in 4.2; fall back to whatever
        ## this build calls it rather than dying on an unknown identifier.
        if engine == 'eevee':
            scene.render.engine = 'BLENDER_EEVEE_NEXT'
        else:
            raise
    return scene.render.engine


def quick_render(camera_obj, resolution_x=64, resolution_y=32, output_path="/tmp/blender_render.png", frame=None, engine=None):
    """
    Renders the scene quickly using low-quality settings for speed.

    The default resolution is deliberately tiny. These renders are the input
    tensor for a small network running on an embedded board, not something a
    person looks at. Note that on EEVEE the resolution barely affects render
    time -- see set_render_engine() -- so the small default is about keeping the
    tensors small, not about speed.
    
    Args:
        camera_obj (bpy.types.Object): The camera object to render from.
        resolution_x (int): Horizontal resolution in pixels.
        resolution_y (int): Vertical resolution in pixels.
        output_path (str): File path for the output PNG image.
        engine (str): Optional render engine for this render onwards.
    """
    if engine is not None:
        set_render_engine(engine)
    if not output_path.startswith('/tmp/'): output_path = '/tmp/' + output_path
    if not output_path.endswith('.png'): output_path += '.png'
    if frame is not None:
        ## Without a frame number every tick overwrites the same file, so an
        ## animation run leaves only its last frame on disk.
        output_path = '%s.%04d.png' % (output_path[:-4], frame)
    scene = bpy.context.scene
    # 1. Set the active camera for the scene
    scene.camera = camera_obj    
    # 2. Configure output resolution and format
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_path    
    # 3. Optimize settings for speed depending on the active render engine
    engine = scene.render.engine
    if engine == 'CYCLES':
        print('USING CYCLES')
        # Minimize Cycles samples and disable heavy features
        scene.cycles.samples = 1
        scene.cycles.preview_samples = 1
        scene.cycles.use_denoising = False
        scene.cycles.max_bounces = 0
        scene.cycles.diffuse_bounces = 0
        scene.cycles.glossy_bounces = 0
        scene.cycles.transmission_bounces = 0
        scene.cycles.volume_bounces = 0        
    elif engine == 'BLENDER_EEVEE' or engine == 'BLENDER_EEVEE_NEXT':
        print('USING EEVEE')
        # Eevee / Eevee Next speed optimizations.
        # NOTE: use_bloom / use_ssr / use_gtao were removed in Blender 4.2 (Eevee Next),
        # so every property is set defensively rather than assumed to exist.
        set_if_exists(scene.eevee, 'taa_render_samples', 1)
        set_if_exists(scene.eevee, 'use_bloom', False)
        set_if_exists(scene.eevee, 'use_ssr', False)
        set_if_exists(scene.eevee, 'use_gtao', False)
        set_if_exists(scene.eevee, 'use_motion_blur', False)
        # Eevee Next equivalents (no-ops on older Blender)
        set_if_exists(scene.eevee, 'use_raytracing', False)
        set_if_exists(scene.eevee, 'use_shadows', False)
        set_if_exists(scene.eevee, 'use_volumetric_shadows', False)
    # 4. Disable anti-aliasing / pixel filter if applicable
    scene.render.film_transparent = False    
    # 5. Execute the render
    print(f"Starting quick render from camera '{camera_obj.name}' to {output_path}...")
    bpy.ops.render.render(write_still=True)
    print("Render complete!")
    return output_path

DEFAULT_ARM = os.path.join(HERE,'abb/irb120/irb120.blend')
RobotSim = None


def normalise_arm_specs(arms):
    """
    Accept the several shapes an arm list may take and return uniform dicts.

        'path.blend'                                    -- default placement
        ('path.blend', (x, y, z))                       -- placed
        ('path.blend', (x, y, z), (rx, ry, rz))         -- placed and rotated
        {'path': ..., 'location': ..., 'rotation': ...,
         'parent': ..., 'name': ...}                    -- everything

    A bare path is still the common case, so it stays the shortest form.
    """
    specs = []
    for entry in (arms or []):
        if isinstance(entry, dict):
            spec = dict(entry)
        elif isinstance(entry, (tuple, list)):
            spec = {'path': entry[0]}
            if len(entry) > 1: spec['location'] = entry[1]
            if len(entry) > 2: spec['rotation'] = entry[2]
        else:
            spec = {'path': entry}
        assert 'path' in spec, 'arm spec has no path: %r' % (entry,)
        assert str(spec['path']).endswith('.blend'), spec['path']
        specs.append(spec)
    return specs


def arm_mount_layout(count, size, index=0):
    """
    Default mount point for arm `index` of `count`, on a body of `size`.

    Arms are spread evenly across the front edge instead of being stacked at one
    point. The single-arm case lands on the centre line, which is exactly where
    the old hard-coded placement put it.
    """
    x, y, z = size
    if count <= 1:
        ox = 0.0
    else:
        ## Interior points of the width, so no arm hangs off the corner.
        ox = -x * 0.5 + x * (index + 1) / float(count + 1)
    return (ox, y * 0.5, z * 0.5 + 0.05)
## ---------------------------------------------------------------------------
## Sensors
## ---------------------------------------------------------------------------
##
## Depth and segmentation are render *passes*, not separate renders: the
## renderer already computes the depth of every pixel it shades, so asking for
## it costs almost nothing on top of the colour image. Measured here, adding
## depth and segmentation to a Cycles render costs +0.004s on a 0.028s render.
## One render per camera therefore produces every modality, and the sensor API
## is shaped around that rather than around one call per sensor.

SENSOR_PASSES = {
    ## name         socket      view-layer flag              file format
    'rgb':          ('Image',   None,                        'PNG',      '.png'),
    'depth':        ('Depth',   'use_pass_z',                'OPEN_EXR', '.exr'),
    'segmentation': ('IndexOB', 'use_pass_object_index',     'OPEN_EXR', '.exr'),
    'normal':       ('Normal',  'use_pass_normal',           'OPEN_EXR', '.exr'),
    'mist':         ('Mist',    'use_pass_mist',             'OPEN_EXR', '.exr'),
}

## EEVEE does not implement the object-index pass -- the socket is not even
## offered on the Render Layers node -- so segmentation forces Cycles.
CYCLES_ONLY_PASSES = ('segmentation',)

## Semantic labels written into the segmentation pass. Background is 0 because
## that is what the renderer leaves where nothing was hit.
SEGMENT_CLASSES = {
    'background': 0,
    'ground': 1,
    'body': 2,
    'wheel': 3,
    'hub': 4,
    'arm': 5,
}


def configure_cycles(scene=None, samples=1, bounces=0):
    """
    Point Cycles at speed rather than beauty.

    One sample with no bounces is flat and noisy, but geometry, depth and
    segmentation are all exact, which is what the passes are for. Raise
    `samples` when the RGB is actually training data rather than a smoke test.
    """
    scene = scene or bpy.context.scene
    if not hasattr(scene, 'cycles'):
        return None
    scene.cycles.samples = samples
    scene.cycles.preview_samples = samples
    ## This build has no OpenImageDenoise; leaving denoising on makes every
    ## render fail outright rather than merely look noisy.
    scene.cycles.use_denoising = False
    for attr in ('max_bounces', 'diffuse_bounces', 'glossy_bounces',
                 'transmission_bounces', 'volume_bounces'):
        set_if_exists(scene.cycles, attr, bounces)
    return scene.cycles


class SensorRig:
    """
    Captures several render passes from one render, via the compositor.

    The node graph is built once and reused. It is only switched on around an
    actual capture: a live File Output node writes files on *every* render in
    the scene, which would otherwise make each plain quick_render() spray pass
    files across the output directory.
    """

    def __init__(self, passes=('rgb',), scene=None, out_dir='/tmp'):
        self.passes = tuple(passes)
        self.scene = scene or bpy.context.scene
        self.out_dir = out_dir
        self._built_for = None      ## (passes, engine) the tree was built for
        self._output_node = None
        self._configured = False

    def __repr__(self):
        return '<SensorRig %s>' % (', '.join(self.passes),)

    # -- setup --------------------------------------------------------------

    def needs_cycles(self):
        return any(p in CYCLES_ONLY_PASSES for p in self.passes)

    def ensure_engine(self):
        """
        Switch to Cycles if any requested pass requires it.

        Configuration happens whenever we end up on Cycles, not only when we
        switch to it: Blender's default is 4096 samples, so a scene that was
        already on Cycles would otherwise render several thousand times slower
        than one this rig switched itself. Done once per rig, so a caller who
        raises the sample count afterwards keeps it.
        """
        if self.needs_cycles() and self.scene.render.engine != 'CYCLES':
            print('sensors: %s requires Cycles; switching from %s' % (
                [p for p in self.passes if p in CYCLES_ONLY_PASSES],
                self.scene.render.engine))
            set_render_engine('cycles')
        if self.scene.render.engine == 'CYCLES' and not self._configured:
            configure_cycles(self.scene)
            self._configured = True
        return self.scene.render.engine

    def enable_passes(self, view_layer=None):
        vl = view_layer or bpy.context.view_layer
        for name in self.passes:
            flag = SENSOR_PASSES[name][1]
            if flag:
                setattr(vl, flag, True)
        return vl

    def build(self):
        """
        (Re)build the compositor graph: Render Layers -> File Output.

        The Render Layers node only exposes a socket once its pass is enabled
        *and* the engine supports it, so the node is recreated here rather than
        reused -- that is also why enable_passes() and ensure_engine() must both
        run first.
        """
        self.ensure_engine()
        self.enable_passes()

        self.scene.use_nodes = True
        tree = self.scene.node_tree
        for node in list(tree.nodes):
            tree.nodes.remove(node)

        layers = tree.nodes.new('CompositorNodeRLayers')
        out = tree.nodes.new('CompositorNodeOutputFile')
        out.base_path = self.out_dir
        out.file_slots.clear()

        for name in self.passes:
            socket, _flag, fmt, _ext = SENSOR_PASSES[name]
            source = layers.outputs.get(socket)
            if source is None or not source.enabled:
                available = [o.name for o in layers.outputs if o.enabled]
                raise RuntimeError(
                    'pass %r (socket %s) is not available on engine %s; '
                    'available: %s' % (name, socket, self.scene.render.engine, available))
            out.file_slots.new(name)
            slot = out.file_slots[-1]
            ## Per-slot format: RGB as PNG so it is viewable, data passes as
            ## 32-bit EXR because they are measurements. Segmentation indices
            ## and metric depth both die in an 8-bit colour-managed PNG.
            slot.use_node_format = False
            slot.format.file_format = fmt
            if fmt == 'OPEN_EXR':
                slot.format.color_depth = '32'
            tree.links.new(source, out.inputs[name])

        self._output_node = out
        self._built_for = (self.passes, self.scene.render.engine)
        self.scene.use_nodes = False    ## dormant until a capture asks for it
        return out

    def ensure_built(self):
        if self._built_for != (self.passes, self.scene.render.engine) or self._output_node is None:
            self.build()
        return self._output_node

    # -- capture ------------------------------------------------------------

    def capture(self, camera, name='capture', frame=None, resolution=None):
        """
        Render `camera` once and write every requested pass.

        Returns {pass_name: path}. `name` prefixes the files, `frame` numbers
        them; without a frame each capture overwrites the last.
        """
        out = self.ensure_built()
        scene = self.scene
        scene.camera = camera
        if resolution:
            scene.render.resolution_x, scene.render.resolution_y = resolution
        scene.render.resolution_percentage = 100

        stem = name if frame is None else '%s.%04d' % (name, frame)
        for slot, pass_name in zip(out.file_slots, self.passes):
            slot.path = '%s.%s.' % (stem, pass_name)

        was_using_nodes = scene.use_nodes
        scene.use_nodes = True
        try:
            bpy.ops.render.render(write_still=False)
        finally:
            scene.use_nodes = was_using_nodes

        ## The File Output node always appends the scene frame to the slot path.
        ## The sim's frame is not the scene's -- during a recording the playhead
        ## deliberately does not move -- so the files are renamed to the names
        ## that were asked for rather than trying to drive scene.frame_current,
        ## which would re-evaluate the animation mid-tick.
        written = {}
        suffix = '%04d' % scene.frame_current
        for pass_name in self.passes:
            ext = SENSOR_PASSES[pass_name][3]
            produced = os.path.join(self.out_dir, '%s.%s.%s%s' % (stem, pass_name, suffix, ext))
            wanted = os.path.join(self.out_dir, '%s.%s%s' % (stem, pass_name, ext))
            if os.path.isfile(produced):
                if os.path.isfile(wanted):
                    os.remove(wanted)
                os.rename(produced, wanted)
            written[pass_name] = wanted
        return written


def read_pass(path, channel=0):
    """
    Read a rendered pass back as (width, height, values).

    Mostly for tests and for sanity-checking a capture; a training pipeline
    would read these straight into numpy. Depth is in metres, with unhit pixels
    at a very large value; segmentation values are the integer pass indices.
    """
    img = bpy.data.images.load(path)
    try:
        w, h = img.size
        px = list(img.pixels)
        return w, h, px[channel::4]
    finally:
        bpy.data.images.remove(img)



CAMERA_SETS = {
    'all':     ['left', 'front', 'right', 'back'],
    'front':   ['front'],
    'forward': ['front'],
    'none':    [],
}
CAMERA_INTERVAL = 30    ## ticks between camera captures, by default


def camera_names(cameras):
    """
    Resolve the `cameras` argument to a list of camera names.

    Accepts a preset name ('all', 'front', 'none'), an explicit list, or None
    for the full four-camera mast.
    """
    if cameras is None:
        return list(CAMERA_SETS['all'])
    if isinstance(cameras, str):
        assert cameras in CAMERA_SETS, 'unknown camera set %r, try %s' % (
            cameras, sorted(CAMERA_SETS))
        return list(CAMERA_SETS[cameras])
    names = list(cameras)
    for n in names:
        assert n in CAMERA_SETS['all'], 'unknown camera %r' % n
    return names


class Robot:
    BOTS = []
    def __init__(self, size=(1,1,0.1), wheels=4, wheel_radius=0.1, arms=[DEFAULT_ARM],
                 drive='differential', cameras='all', camera_interval=CAMERA_INTERVAL,
                 passes=('rgb',), out_dir='/tmp',
                 max_accel=None, max_yaw_accel=None):
        Robot.BOTS.append(self)
        if RobotSim: RobotSim.bots.append(self)
        self.size = size
        self.wheel_radius = wheel_radius
        ## Ticks between camera captures. Vision is the expensive sense: on the
        ## target hardware the base loop runs on cheap fast sensors and only
        ## peeks at the cameras periodically, so the sim models the same duty
        ## cycle rather than rendering every tick.
        self.camera_interval = camera_interval
        self.last_capture_tick = None
        self.captures = 0
        self.out_dir = out_dir
        ## One rig per robot, capturing every requested modality in a single
        ## render per camera.
        self.sensors = SensorRig(passes, out_dir=out_dir)
        self.lidars = []
        self.contact = None      ## set by enable_contact()
        self.boards = []         ## every MCU on this robot
        self.bindings = []       ## the ones wired to the base, see attach_firmware()
        self._network = None     ## lazily created; this robot's internal bus
        x,y,z = size

        self.root = create_empty('ROBOT.ROOT')
        self.body = create_cube('ROBOT.BODY', size )
        self.body.parent = self.root

        ## -- wheels ---------------------------------------------------------
        ## `wheels` is either a count (laid out automatically) or an explicit
        ## list of specs, so a caller can place every wheel by hand.
        self.wheels = []     ## objects, in layout order
        self.wheel_map = {}  ## name -> object, the original interface
        self.wheel_list = [] ## Wheel records, what the drive model actually uses
        specs = wheels if isinstance(wheels, (list, tuple)) else wheel_layout(wheels, size)
        specs = [self._normalise_wheel_spec(s, i, wheel_radius, size)
                 for i, s in enumerate(specs)]
        for spec in specs:
            obj = create_cylinder(spec['name'], radius=spec['radius'],
                                  depth=spec['radius'], location=spec['location'])
            obj.parent = self.root
            self.wheels.append(obj)
            self.wheel_map[obj.name] = obj
            self.wheel_list.append(Wheel(
                obj, name=obj.name, x=spec['x'], y=spec['y'],
                side=spec['side'], axle=spec['axle'],
                driven=spec.get('driven', True), steerable=spec.get('steerable'),
                radius=spec['radius']))

        ## Base motion model. Kinematic for now; an external physics engine can
        ## replace step() later while keeping this command interface.
        ## Track and wheelbase are measured from the wheels that actually exist,
        ## falling back to the body box when the layout cannot supply them (a
        ## single wheel has no track, one axle has no wheelbase).
        model = DRIVE_MODELS.get(drive, DifferentialDrive)
        ## Acceleration limits are how a base gets momentum. Left unset the
        ## robot reaches its commanded speed within one tick, as before.
        kw = {'track': layout_track(specs, fallback=x),
              'max_accel': max_accel, 'max_yaw_accel': max_yaw_accel}
        if issubclass(model, AckermannDrive):
            kw['wheelbase'] = layout_wheelbase(specs, fallback=y)
        self.drive = model(self.root, wheels=self.wheel_list,
                           wheel_radius=wheel_radius, **kw)

        self.root.location.z = wheel_radius * 1.5

        ## -- hubs and cameras -----------------------------------------------
        ## Built before the arms so that an arm can be mounted onto one.
        self.front_hub = create_cube('FRONT.HUB', size=(0.4,0.2,0.1), location=(0,y/2,0.05))
        self.front_hub.parent = self.root

        self.rear_hub = create_cube('REAR.HUB', size=(0.4,0.2,0.1), location=(0,-y/2,0.05))
        self.rear_hub.parent = self.root

        self.camera_hub = create_cube('CAM.HUB', size=(0.18,0.18,0.5))
        self.camera_hub.location.y = -y / 2
        self.camera_hub.location.z = 0.25
        self.camera_hub.parent = self.root

        self.cameras = { k : create_camera(k) for k in camera_names(cameras) }
        for cam in self.cameras.values():
            cam.parent = self.camera_hub
            cam.location.z = 0.3

        ## -- arms -----------------------------------------------------------
        self.arms = []       ## Arm objects (joint-space control)
        self.arm_roots = []  ## the ARM.ROOT empties, as before
        self.mounts = []     ## the spec each arm was built from
        arm_specs = normalise_arm_specs(arms)
        for i, spec in enumerate(arm_specs):
            spec.setdefault('location', arm_mount_layout(len(arm_specs), size, i))
            self.add_arm(**spec)

        self.label_parts()

    def label_parts(self, offset=0):
        """
        Write semantic class indices onto every part, for the segmentation pass.

        `offset` shifts this robot's labels, which is how instances are told
        apart: two robots labelled with different offsets segment separately
        while keeping their class structure. Objects the robot does not own --
        the ground, the world -- keep index 0 and read as background.
        """
        groups = [
            ('body', [self.body]),
            ('wheel', self.wheels),
            ('hub', [self.front_hub, self.rear_hub, self.camera_hub]),
            ('arm', [ob for root in self.arm_roots
                     for ob in self._descendants(root) if ob.type == 'MESH']),
        ]
        for class_name, objects in groups:
            index = SEGMENT_CLASSES[class_name] + offset
            for ob in objects:
                if ob is not None:
                    ob.pass_index = index
        self.segment_offset = offset
        return self

    @staticmethod
    def _descendants(obj):
        """Every object under `obj`, including nested arm links."""
        out = []
        stack = list(obj.children)
        while stack:
            ob = stack.pop()
            out.append(ob)
            stack.extend(ob.children)
        return out

    def _normalise_wheel_spec(self, spec, index, wheel_radius, size):
        """
        Fill in whatever a wheel spec left out.

        Custom layouts should only have to say where the wheel goes; the name,
        side, axle and radius all have sensible answers derivable from that.
        """
        if not isinstance(spec, dict):
            spec = {'location': tuple(spec)}
        else:
            spec = dict(spec)
        loc = tuple(spec.get('location', (0.0, 0.0, -size[2] * 0.5)))
        spec['location'] = loc
        spec.setdefault('x', loc[0])
        spec.setdefault('y', loc[1])
        ## Side follows the sign of the lateral offset, so a hand-placed wheel
        ## still steers and rolls as the correct side of the vehicle.
        spec.setdefault('side', 'C' if abs(spec['x']) < 1e-9 else ('L' if spec['x'] < 0 else 'R'))
        spec.setdefault('axle', 'MID')
        spec.setdefault('name', 'W.%s.%s' % (spec['side'], spec['axle']))
        spec.setdefault('radius', wheel_radius)
        return spec

    def add_arm(self, path, location=None, rotation=None, parent=None, name='ARM.ROOT'):
        """
        Load a .blend arm and mount it on this robot.

        `location` and `rotation` are relative to `parent`, which may be an
        object or the name of one of this robot's parts ('root', 'body',
        'front_hub', 'rear_hub', 'camera_hub'). Defaults reproduce the original
        placement: centre of the front edge, rotated to face forward.
        """
        assert str(path).endswith('.blend'), path
        loaded = load_blend_objects(path)
        print(loaded)
        root = create_empty(name)
        tip = None
        for ob in loaded:
            if ob.name.startswith("Tool tip"): tip = ob
            if not ob.parent: ob.parent = root
        assert tip, 'no "Tool tip" object found in %s' % path
        tip.parent = root

        if location is None:
            location = arm_mount_layout(1, self.size, 0)
        root.location = location
        ## The arm rigs are authored facing +X, so the default quarter turn puts
        ## the tool tip out the front of the robot (+Y).
        root.rotation_euler = rotation if rotation is not None else (0.0, 0.0, math.pi / 2)

        root.parent = self.resolve_part(parent)
        self.arm_roots.append(root)
        self.mounts.append({'path': path, 'location': tuple(location),
                            'rotation': tuple(root.rotation_euler),
                            'parent': root.parent.name if root.parent else None})
        built = []
        for arm in Arm.from_objects(loaded, root=root):
            print('ARM:', arm, arm.names)
            self.arms.append(arm)
            built.append(arm)
        ## Parenting only reaches matrix_world on the next depsgraph evaluation,
        ## so without this a caller reading the mount point straight after
        ## add_arm() gets the pre-parent transform -- the same stale-read footgun
        ## kinematics.Arm.update() exists to avoid.
        bpy.context.view_layer.update()
        return built

    def resolve_part(self, part):
        """Turn a part name into the object it refers to; None means the root."""
        if part is None or part == 'root':
            return self.root
        if isinstance(part, str):
            obj = getattr(self, part, None)
            assert obj is not None, 'unknown robot part: %r' % part
            return obj
        return part

    @property
    def network(self):
        """This robot's internal bus, created on first use."""
        if self._network is None:
            self._network = Network(name='%s.bus' % self.root.name, owner=self)
            for board in self.boards:
                self._network.add(board)
        return self._network

    def step(self, dt):
        """
        Advance this robot by one timestep.

        With firmware attached the order within a tick matters: each board's
        last command is applied, then the plant moves, then what the plant did
        is sensed back, then every board is given the same dt, and only once
        they have all reached the same virtual time are messages routed.

        Sensing after the plant lets firmware see the consequence of its own
        command on the tick it lands rather than one tick late. Delivering after
        every board has stepped is what gives messages their one-step latency
        and keeps the result independent of the order boards are listed in.
        """
        for binding in self.bindings:
            binding.apply()
        if self.drive:
            self.drive.step(dt)
        for binding in self.bindings:
            binding.sense()
        for board in self.boards:
            board.step(dt)
        ## Only deliver a bus this robot owns. One spanning several robots is
        ## delivered once they have all stepped -- see RobotSim.update().
        if self._network is not None and self._network.owner is self:
            self._network.deliver()

    def attach_board(self, source, name=None, echo=False, build_args=None,
                     loop_hz=1000):
        """
        Add an MCU that is not wired to the base.

        A navigation planner, a vision node, an arm controller: boards that
        think and talk but do not drive wheels. Use attach_firmware() for one
        that does.
        """
        if not firmware.available():
            raise RuntimeError(firmware.why_unavailable())
        so = firmware.build(source, **(build_args or {}))
        board = Board(so, name=name or os.path.basename(source), echo=echo,
                      loop_hz=loop_hz)
        board.start()
        self.boards.append(board)
        if self._network is not None:
            self._network.add(board)
        else:
            self.network.add(board)
        return board

    def attach_firmware(self, source, name=None, target=None, echo=False,
                        counts_per_metre=1000.0, top_speed=None,
                        source_mode='wheel', build_args=None, loop_hz=1000, **kw):
        """
        Compile firmware and wire it to this robot's base.

        `source` is a .c or .cpp file. C++ is lowered through crust's C++ subset
        front end first, which refuses what it cannot lower rather than guessing.

        Returns the FirmwareDrive binding; its `.board` is the MCU, which is
        what you set a target on and read the console from. Needs crust cloned
        beside robotsim -- see firmware.available().
        """
        if not firmware.available():
            raise RuntimeError(firmware.why_unavailable())
        so = firmware.build(source, **(build_args or {}))
        board = Board(so, name=name or os.path.basename(source), echo=echo,
                      loop_hz=loop_hz)
        binding = FirmwareDrive(board, self, counts_per_metre=counts_per_metre,
                                top_speed=top_speed, source=source_mode, **kw)
        board.start()
        if target is not None:
            board.target = target
        self.boards.append(board)
        self.bindings.append(binding)
        self.network.add(board)
        return binding

    def firmware_console(self):
        """Everything every board on this robot has printed."""
        return {b.name: b.console for b in self.boards}

    def stop(self):
        if self.drive:
            self.drive.stop()
        return self

    def render_cameras(self, frame=None, resolution=None):
        """
        Render every camera on this robot, right now, unconditionally.

        Use sample_cameras() instead to respect the capture interval.
        """
        kw = {}
        if resolution:
            kw['resolution_x'], kw['resolution_y'] = resolution
        paths = []
        for cam in self.cameras.values():
            paths.append(
                quick_render(cam, output_path='.'.join( [self.root.name, cam.name]),
                             frame=frame, **kw)
            )
        return paths

    def camera_due(self, tick=None):
        """
        Is this tick a capture tick?

        The first tick always captures -- a controller needs an initial view
        before it has anything to reason about. After that, captures are spaced
        `camera_interval` ticks apart. An interval of 0 disables capture
        entirely; 1 captures every tick.
        """
        if not self.cameras or not self.camera_interval:
            return False
        if tick is None:
            tick = RobotSim.ticks if RobotSim else 0
        if self.last_capture_tick is None:
            return True
        return (tick - self.last_capture_tick) >= self.camera_interval

    def enable_contact(self, ground=True, collide=True, level=False,
                       backend='ray', **kw):
        """
        Give this robot's drive model a contact model.

        Without one the commanded pose is the pose: the base hovers at a fixed
        height and drives through walls. With one it rides the ground, stops at
        obstacles, and reports slip.

        Contact points default to the wheels, which is what makes ground
        following and levelling agree with where the robot actually touches.

        `backend` picks which kind of answer the world gives back:

        'ray'    -- contact.RayContact, the default. Kinematic: a handful of
                    ray casts per tick, no mass, and slip is reported rather
                    than suffered. Fast enough to generate a corpus with.
        'mujoco' -- muble.MujocoContact. Rigid-body dynamics: the robot has
                    weight, momentum survives a collision, and a wheel asked
                    for more grip than the surface has will spin instead.
                    Needs `pip install mujoco`.

        Both satisfy `drive.ContactModel`, so nothing downstream changes --
        which is the whole point of the seam. Switch backends to find out how
        much of a policy's behaviour was resting on contact being free.
        """
        points = kw.pop('contact_points', None)
        if points is None:
            points = [(w.x, w.y) for w in self.wheel_list] or [(0.0, 0.0)]

        if backend == 'mujoco':
            self.contact = self.mujoco_contact(points, ground=ground, **kw)
        elif backend == 'ray':
            kw.setdefault('radius', max(self.size[0], self.size[1]) * 0.5)
            kw.setdefault('ride_height', self.wheel_radius * 1.5)
            kw.setdefault('ignore', self.parts())
            self.contact = RayContact(ground=ground, collide=collide,
                                      level=level, contact_points=points, **kw)
        else:
            raise ValueError("backend must be 'ray' or 'mujoco', not %r" % backend)
        self.drive.contact = self.contact
        if ground:
            ## Start grounded rather than wherever the caller happened to leave
            ## the robot, so the first step is a normal step.
            bpy.context.view_layer.update()
            self.contact.snap(self.root)
        return self.contact

    def mujoco_contact(self, points, ground=True, objects=None, **kw):
        """
        Build a MuJoCo world from the current scene and wrap it as a contact model.

        Imported here rather than at module scope so MuJoCo stays genuinely
        optional: a robotsim install without it behaves exactly as before and
        only a caller that asks for the dynamic backend ever pays for it.

        Static geometry comes from the scene's meshes minus this robot's own
        parts -- without that exclusion the robot is built into the world it is
        supposed to be driving through, and it starts the run wedged inside
        itself.
        """
        try:
            import muble
        except ImportError as exc:
            raise ImportError(
                "backend='mujoco' needs MuJoCo: pip install mujoco. "
                "The default backend='ray' has no extra dependencies.") from exc

        kw.setdefault('mass', getattr(self, 'mass', 10.0))
        kw.setdefault('size', tuple(self.size))
        kw.setdefault('wheel_radius', self.wheel_radius)
        ## The scene's floor is a mesh like any other and comes across with the
        ## rest of the geometry, so no plane is added unless one is asked for.
        ## Adding one by default would put an invisible surface at z=0 through
        ## the middle of any scene whose ground sits lower.
        kw.setdefault('ground_z', None)
        bpy.context.view_layer.update()
        return muble.MujocoContact.from_blender(
            robot=self, objects=objects, contact_points=points, **kw)

    def add_lidar(self, location=None, rotation=None, parent=None, name='LIDAR',
                  self_filter=True, **kw):
        """
        Mount a lidar on this robot.

        Defaults put it on top of the camera mast, where a real scanner goes: it
        needs a clear horizon, and anything below the body line spends most of
        its beams on the robot's own wheels.

        `self_filter` makes the beams pass through this robot's own parts, which
        is what a real unit's masking does. Turn it off to see the robot in its
        own scan, which is occasionally useful for checking a mount position.
        """
        mount = create_empty(name)
        if location is None:
            ## Just above the camera mast: mast half-height plus clearance.
            location = (0.0, 0.0, 0.3)
            parent = 'camera_hub' if parent is None else parent
        mount.location = location
        if rotation is not None:
            mount.rotation_euler = rotation
        mount.parent = self.resolve_part(parent)
        bpy.context.view_layer.update()

        ignore = set(kw.pop('ignore', ()))
        if self_filter:
            ignore |= set(self.parts())
        lidar = Lidar(mount, ignore=ignore, **kw)
        self.lidars.append(lidar)
        return lidar

    def parts(self):
        """Every object this robot owns, for self-filtering and labelling."""
        out = [self.root, self.body, self.front_hub, self.rear_hub, self.camera_hub]
        out += list(self.wheels)
        out += list(self.cameras.values())
        for root in self.arm_roots:
            out.append(root)
            out += self._descendants(root)
        for lidar in getattr(self, 'lidars', []):
            out.append(lidar.mount)
        return [o for o in out if o is not None]

    def scan_lidars(self, tick=None, force=False):
        """
        Sample every lidar that is due this tick.

        Returns {index: LidarScan} for the ones that fired. Lidars default to
        every tick, so unlike the cameras this normally returns them all -- that
        asymmetry is the point.
        """
        if tick is None:
            tick = RobotSim.ticks if RobotSim else 0
        depsgraph = bpy.context.evaluated_depsgraph_get()
        scene = bpy.context.scene
        out = {}
        for i, lidar in enumerate(self.lidars):
            scan = lidar.sample(tick=tick, force=force, depsgraph=depsgraph, scene=scene)
            if scan is not None:
                out[i] = scan
        return out

    def sample_cameras(self, frame=None, tick=None, force=False, resolution=None):
        """
        Render the cameras if this tick is due, otherwise do nothing.

        Returns the list of PNG paths, or an empty list when it was not time.
        This is the call a control loop should make every tick: it keeps the
        expensive sense on its own slower clock while the base loop runs at full
        rate on the cheap ones.
        """
        if not self.capture_due(tick, force):
            return []
        return self.render_cameras(frame=frame, resolution=resolution)

    # -- multi-pass sensors -------------------------------------------------

    def capture_due(self, tick=None, force=False):
        """Is a capture due this tick? Marks it taken when it is."""
        if tick is None:
            tick = RobotSim.ticks if RobotSim else 0
        if not force and not self.camera_due(tick):
            return False
        self.last_capture_tick = tick
        self.captures += 1
        return True

    def capture(self, frame=None, passes=None, resolution=None):
        """
        Capture every requested pass from every camera, unconditionally.

        Returns {camera_name: {pass_name: path}}. Each camera costs exactly one
        render no matter how many passes are asked for, because the passes all
        come out of that single render.
        """
        if passes is not None:
            self.sensors.passes = tuple(passes)
        out = {}
        for name, cam in self.cameras.items():
            out[name] = self.sensors.capture(
                cam, name='.'.join([self.root.name, name]),
                frame=frame, resolution=resolution)
        return out

    def sample(self, frame=None, tick=None, force=False, passes=None, resolution=None):
        """
        Rate-limited multi-pass capture: the sensor-rig equivalent of
        sample_cameras(). Returns {} when this tick is not a capture tick.
        """
        if not self.capture_due(tick, force):
            return {}
        return self.capture(frame=frame, passes=passes, resolution=resolution)

class RobotSimpleSim:
    """
    Fixed-timestep loop. `dt` is simulated time per tick, not wall-clock: the
    loop runs as fast as it can and the physics/motion advance by exactly `dt`
    each tick, so results are reproducible regardless of how slow rendering is.
    """
    def __init__(self, dt=1.0/30.0):
        self.callbacks = []
        self.bots = []
        self.ticks = 0
        self.dt = dt
        self.time = 0.0          ## simulated seconds elapsed
        self.step_bots = True    ## set False to drive bots manually
        ## Buses spanning several robots. A robot's own bus is delivered by its
        ## own step(); a shared one cannot be, because delivering from inside
        ## one robot's step would route before the others had caught up.
        self.networks = []
        self.recorder = None     ## set via record(); bakes ticks onto the timeline
        self._in_update = False
        self._finish_pending = False

    def __call__(self, cb):
        self.callbacks.append(cb)
        return cb

    def record(self, **kw):
        """Start baking each tick onto the Blender timeline as keyframes."""
        self.recorder = Recorder(self, **kw)
        return self.recorder

    @property
    def frame(self):
        """Current scene frame, if recording."""
        return self.recorder.frame if self.recorder else None

    def update(self, dt=None):
        if dt is None: dt = self.dt
        self._in_update = True
        try:
            ## Bots step first so callbacks observe the post-step state.
            if self.step_bots:
                for bot in self.bots: bot.step(dt)
            ## Buses spanning several robots are delivered here, once every
            ## board on every robot has reached the same virtual time.
            for net in self.networks: net.deliver()
            for cb in self.callbacks:
                ## Callbacks may take (dt) or no arguments; both styles are supported
                ## so existing zero-arg callbacks keep working.
                try:
                    nargs = cb.__code__.co_argcount
                except AttributeError:
                    nargs = 0
                cb(dt) if nargs else cb()
            ## Capture after callbacks so the keyframe reflects the final state of
            ## the tick, including anything the callback changed.
            if self.recorder: self.recorder.capture()
            self.ticks += 1
            self.time += dt
        finally:
            self._in_update = False
        ## Only now is it safe to finish: finish() rewinds the scene, which
        ## re-evaluates the animation and moves everything back to the start
        ## frame. Doing that before the capture above would bake the rewound
        ## pose into this tick's keyframe.
        if self._finish_pending:
            self._finish_pending = False
            if self.recorder: self.recorder.finish()

    def stop(self):
        self.callbacks = []
        if self.recorder:
            ## A callback calling stop() is still mid-tick, so defer the rewind
            ## until update() has captured this tick.
            if self._in_update: self._finish_pending = True
            else: self.recorder.finish()
    

def main():
    global RobotSim
    RobotSim = RobotSimpleSim()
    for arg in script_args:
        if arg.endswith('.blend'):
            loaded = load_blend_objects(arg)
            print(loaded)
        elif arg.endswith('.py'):
            py = open(arg).read()
            print('exec:',arg)
            exec(py, globals(), globals() )

    while RobotSim.callbacks:
        RobotSim.update()

if __name__=='__main__':
    main()
