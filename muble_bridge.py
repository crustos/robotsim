"""
Read MuBlE scenes into robotsim.

`muble.py` borrows MuBlE's physics. This borrows its *scenes* -- the tabletop
layouts of mugs, glasses, boards and bowls that MuBlE generates for manipulation
benchmarks -- and puts them in front of robotsim's four render passes. The result
is line art, object index and metric depth for scenes robotsim's own randomiser
would never produce, which is exactly the appearance variation Stage 1 is
supposed to absorb.

The handoff file is written by `robotsim_export.py` on the MuBlE side. It is
plain JSON with no MuBlE, MuJoCo or robosuite import anywhere in the path, so a
corpus can be generated from MuBlE scenes on a machine that has none of them
installed. Raw MuBlE `scenes.json` bundles are also accepted directly, for the
case where the exporter has not been run.

    import muble_bridge
    scene = muble_bridge.load('scene_000000.json')
    objects, labels = muble_bridge.to_blender(scene)   ## needs bpy
    contact = muble_bridge.build_contact(scene)        ## needs mujoco

Geometry is MuBlE's own. Rendering appends the authored `.blend` for each object
from MuBlE's shape library, so the RGB pass shows the same meshes and materials
MuBlE renders; physics uses the shipped convex-hull decomposition, so the robot
collides with a mug's handle rather than with the box around it. Both fall back
to the bounding box per object when an asset is missing, so a partial checkout
degrades rather than fails.

Neither dependency is imported at module scope: the file is readable, its labels
are enumerable, and its geometry is inspectable with nothing installed at all.
"""

import json
import math
import os


FORMAT = 'robotsim/muble-handoff'

## robotsim's own SEGMENT_CLASSES occupy 0-5 and its procedural obstacles use 7,
## so imported objects start well clear of both. Sharing an index with 'ground'
## or 'obstacle' would not error -- it would quietly train the semantic head to
## call a mug a floor.
LABEL_BASE = 16


def load(path, root=None):
    """
    Read a handoff file, or a raw MuBlE scenes bundle.

    Converting a raw bundle here rather than refusing it means the bridge still
    works when someone points it at MuBlE's output directly, which is what they
    will try first. A raw bundle carries no asset paths, so `root` -- a MuBlE
    checkout -- is what lets the converted scene reach real geometry; without it
    the scene still loads and falls back to boxes.
    """
    with open(path) as handle:
        data = json.load(handle)

    if isinstance(data, dict) and data.get('format') == FORMAT:
        if root:
            data.setdefault('assets', {})['root'] = root
        return data

    ## A raw MuBlE bundle: convert the first scene using the exporter's own
    ## rules, imported from wherever MuBlE happens to be checked out.
    scenes = data['scenes'] if isinstance(data, dict) and 'scenes' in data else data
    if not isinstance(scenes, list) or not scenes:
        raise ValueError('%s is neither a handoff nor a MuBlE scenes file' % path)
    return convert(scenes[0], root=root)


## Where MuBlE keeps its assets, relative to a checkout. Mirrors ASSET_DIRS in
## robotsim_export.py; duplicated for the same reason `convert` is, below.
ASSET_DIRS = {
    'shapes': 'scene_generation/data/shapes',
    'materials': 'scene_generation/data/materials',
    'meshes': 'meshes/train',
}


def hulls_for(shape, root):
    """MuBlE's convex decomposition for one shape, numerically ordered."""
    if not root:
        return []
    directory = os.path.join(root, ASSET_DIRS['meshes'], shape)
    if not os.path.isdir(directory):
        return []
    prefix = '%s_hull_' % shape
    names = [f for f in os.listdir(directory)
             if f.startswith(prefix) and f.endswith('.stl')]

    def order(name):
        stem = name[len(prefix):-len('.stl')]
        return int(stem) if stem.isdigit() else 0

    return [os.path.join(ASSET_DIRS['meshes'], shape, n)
            for n in sorted(names, key=order)]


def convert(scene, index=0, root=None):
    """
    Turn one raw MuBlE scene dict into a handoff, without importing MuBlE.

    Duplicates a little of `robotsim_export.export_scene` on purpose. The two
    repositories are separate checkouts and neither can import the other, so the
    alternative to a small duplication is a hard dependency in both directions.

    What it cannot duplicate is the material-override decision: that needs
    MuBlE's object property tables, which live in the MuBlE checkout. A raw
    bundle converted here therefore renders every object with its authored
    appearance. Run `robotsim_export.py` for scenes whose colours were
    randomised.
    """
    objects = []
    for i, obj in enumerate(scene.get('objects', [])):
        bbox = obj.get('bbox')
        if not bbox:
            continue
        x, y, z = obj['3d_coords']
        ex, ey, ez = bbox['x'], bbox['y'], bbox['z']
        quat = obj.get('orientation')
        if quat:
            w, qx, qy, qz = quat
            yaw = math.atan2(2.0 * (w * qz + qx * qy),
                             1.0 - 2.0 * (qy * qy + qz * qz))
        else:
            yaw = math.radians(obj.get('rotation', 0.0))
        shape = obj.get('file', 'object')
        objects.append({
            'id': i + 1,
            'name': '%s_%02d' % (shape, i),
            'label': obj.get('name', shape),
            'shape': shape,
            ## Two placements, not interchangeable: `origin` is MuBlE's own
            ## 3d_coords and is where the real mesh goes; `position` is the
            ## bounding box centre, half an object higher, and belongs to the
            ## box fallback only.
            'origin': [x, y, z],
            'position': [x, y, z + ez * 0.5],
            'size': [ex, ey, ez],
            'scale': obj.get('scale_factor', 1.0),
            'quaternion': list(quat) if quat else list(_yaw_quat(yaw)),
            'yaw': yaw,
            'material_override': None,
            'collision': hulls_for(shape, root),
            'movable': obj.get('movability') != 'fixed',
            'mass': (obj['weight_gt'] / 1000.0
                     if obj.get('weight_gt') is not None else None),
            'material': obj.get('material'),
            'colour': obj.get('colour'),
        })
    out = {'format': FORMAT, 'version': 2,
           'index': scene.get('image_index', index),
           'source': scene.get('image_filename'), 'objects': objects}
    if root:
        out['assets'] = dict(ASSET_DIRS, root=root)
    return out


def _yaw_quat(yaw):
    return (math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5))


def labels(scene, base=LABEL_BASE):
    """
    The token -> pass_index map this scene needs recorded in the manifest.

    The paper's semantic bridge is written per sample rather than once per
    corpus, which is what keeps a corpus readable when its labelling changes
    midway. This returns the half of that map the imported objects contribute.

    Keyed by the object's *label* rather than its unique name, so two mugs share
    a class -- which is what a semantic map is for. The index is taken from the
    first object carrying that label.
    """
    out = {}
    for obj in scene.get('objects', []):
        token = obj.get('label') or obj['name']
        out.setdefault(token, base + obj['id'])
    if 'table' in scene:
        out.setdefault('table', base)
    return out


def label_of(obj, base=LABEL_BASE):
    return base + obj['id']


def asset_paths(scene, root=None):
    """
    Absolute locations of MuBlE's shape, material and mesh directories.

    `root` overrides whatever the handoff recorded, which is what makes a corpus
    file generated on one machine usable on another: the relative layout inside
    a MuBlE checkout is stable, the checkout's location is not.
    """
    assets = dict(scene.get('assets') or {})
    base = root or assets.get('root')
    if not base:
        return {}
    return {key: os.path.join(base, assets[key])
            for key in ('shapes', 'materials', 'meshes') if key in assets}


def resolve(scene, relative, root=None):
    """A path recorded in the handoff, made absolute against the asset root."""
    if not relative:
        return None
    base = root or (scene.get('assets') or {}).get('root')
    if not base:
        return None
    path = os.path.join(base, relative)
    return path if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------

def build_contact(scene, table=True, root=None, hulls=True, **kw):
    """
    A MujocoContact whose world is this scene's geometry.

    Imported lazily, so reading and inspecting a scene needs no MuJoCo.
    """
    import muble
    contact = muble.MujocoContact(**kw)
    add_statics(scene, contact, table=table, root=root, hulls=hulls)
    return contact.build()


def add_statics(scene, contact, table=True, root=None, hulls=True):
    """
    Add this scene's geometry to an existing, unbuilt MujocoContact.

    With `hulls` on and the meshes available, each object goes in as MuBlE's own
    convex decomposition -- one MuJoCo mesh geom per hull STL. A mesh geom in
    MuJoCo is convex regardless, so a decomposition maps across exactly, and the
    robot collides with the real shape of a mug handle rather than with the
    bounding box around it.

    Falls back to the bounding box per object, not per scene, so one missing
    asset costs one approximate object rather than the whole world.
    """
    if table and 'table' in scene:
        top = scene['table']
        contact.add_box(top['name'], top['position'], top['size'])
    elif table:
        ## MuBlE scenes sit on a table at z=0 whose extent is not recorded. With
        ## no surface at all every object falls forever, so a plane is the safe
        ## default -- and it is the same z the objects were placed against.
        contact.add_ground(z=0.0, name='TABLE')

    for obj in scene.get('objects', []):
        paths = [resolve(scene, p, root) for p in obj.get('collision') or []]
        paths = [p for p in paths if p]
        if hulls and paths:
            for i, path in enumerate(paths):
                contact.add_mesh_file(
                    '%s_h%d' % (obj['name'], i), path,
                    ## The hulls are authored about the object's own origin, so
                    ## they are placed at `origin` -- the mid-bottom -- and not
                    ## at the bounding box centre.
                    pos=obj.get('origin', obj['position']),
                    quat=obj.get('quaternion'),
                    scale=obj.get('scale', 1.0))
        else:
            contact.add_box(obj['name'], obj['position'], obj['size'],
                            yaw=obj.get('yaw', 0.0))
    return contact


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def to_blender(scene, base=LABEL_BASE, table=True, collection=None,
               root=None, meshes=True):
    """
    Bring this scene into Blender, labelled for the object-index pass.

    With `meshes` on, each object is appended from MuBlE's own
    `scene_generation/data/shapes/<shape>.blend` using the same protocol MuBlE's
    `add_object_quaternion` uses, and carries the materials it was authored
    with. That is what makes the RGB pass match MuBlE's render rather than
    approximate it: the geometry, the shading and the placement are all the
    originals, not a reconstruction.

    Objects whose appearance MuBlE randomises get the override applied the same
    way MuBlE applies it -- see `_override_material`. Objects whose appearance is
    fixed are left exactly as authored.

    Falls back to a bounding box per object when an asset is missing, so a
    partial checkout degrades instead of failing.

    Returns (objects, labels) where `labels` is the token -> index map to write
    into the dataset manifest.
    """
    import bpy

    paths = asset_paths(scene, root)
    shapes = paths.get('shapes') if meshes else None
    loaded_materials = False

    created = []
    if table and 'table' in scene:
        top = scene['table']
        obj = _cube(bpy, top['name'], top['position'], top['size'])
        obj.pass_index = base
        created.append(obj)

    for spec in scene.get('objects', []):
        obj = None
        if shapes:
            obj = _append_shape(bpy, shapes, spec)

        if obj is not None:
            override = spec.get('material_override')
            if override:
                if not loaded_materials and paths.get('materials'):
                    _load_material_groups(bpy, paths['materials'])
                    loaded_materials = True
                _override_material(bpy, obj, override)
        else:
            ## No asset: the bounding box is the honest approximation, and it
            ## still gives correct depth, segmentation and line art for a solid
            ## of that size.
            obj = _cube(bpy, spec['name'], spec['position'], spec['size'],
                        yaw=spec.get('yaw', 0.0))
            _paint(bpy, obj, spec)

        ## The whole reason the bridge carries integer ids: this single
        ## assignment is what makes the object identifiable in the semantic map.
        obj.pass_index = label_of(spec, base)
        ## The category, in words. The index survives into the semantic map and
        ## the word survives into the caption; without this the description
        ## would have to call a mug 'mug2_00'.
        obj['label'] = spec.get('label') or spec['name']
        created.append(obj)

    if collection is not None:
        for obj in created:
            for old in list(obj.users_collection):
                old.objects.unlink(obj)
            collection.objects.link(obj)

    return created, labels(scene, base)


def _append_shape(bpy, shapes_dir, spec):
    """
    Append one object from MuBlE's shape library and place it.

    Two details that are each silently fatal.

    First, the append is issued as `directory` plus `filename` rather than as
    one combined path. MuBlE passes `filename=<dir>/<name>.blend/Object/<name>`,
    which works only because its `object_dir` is relative: given an absolute
    path Blender drops the leading separator, reports "nothing indicated", and
    returns success with nothing appended. The two-argument form is
    path-agnostic.

    Second, the new datablock is found by diffing the object table across the
    append rather than by looking up the shape's name. MuBlE looks it up
    directly, which works only because it renames every object the instant it
    arrives; if anything in the scene already holds that name, Blender suffixes
    the incoming object and the lookup silently returns the *previous* one,
    which then gets moved to the new object's pose while the new object stays at
    the origin.
    """
    path = os.path.join(shapes_dir, '%s.blend' % spec['shape'])
    if not os.path.isfile(path):
        return None

    directory = os.path.join(path, 'Object') + os.sep
    before = set(bpy.data.objects.keys())
    try:
        bpy.ops.wm.append(filepath=os.path.join(directory, spec['shape']),
                          directory=directory, filename=spec['shape'])
    except RuntimeError:
        return None
    arrived = set(bpy.data.objects.keys()) - before
    if not arrived:
        ## Blender reports a failed append as an error message rather than an
        ## exception, so an empty diff is the only reliable signal.
        return None

    obj = bpy.data.objects[sorted(arrived)[0]]
    obj.name = spec['name']

    scale = spec.get('scale', 1.0)
    obj.scale = (scale, scale, scale)
    quat = spec.get('quaternion')
    if quat:
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = tuple(quat)
    else:
        obj.rotation_euler = (0.0, 0.0, spec.get('yaw', 0.0))
    ## `origin` is MuBlE's 3d_coords, which is where MuBlE itself puts the
    ## object. `position` is the bounding box centre and belongs to the box
    ## fallback only.
    obj.location = tuple(spec.get('origin', spec['position']))
    return obj


def _load_material_groups(bpy, materials_dir):
    """
    Append MuBlE's material node groups, once per session.

    Each .blend in the directory holds a single NodeTree of the same name, which
    is what `_override_material` instances.
    """
    if not os.path.isdir(materials_dir):
        return []
    loaded = []
    for name in sorted(os.listdir(materials_dir)):
        if not name.endswith('.blend'):
            continue
        stem = os.path.splitext(name)[0]
        if stem in bpy.data.node_groups:
            loaded.append(stem)
            continue
        ## Same two-argument form as _append_shape, and for the same reason: the
        ## combined-path form silently appends nothing when given an absolute
        ## path, and a missing node group here does not raise -- it produces a
        ## blank override material, which replaces MuBlE's shader with grey
        ## while every check that only looks for "was it overridden" still
        ## passes.
        directory = os.path.join(materials_dir, name, 'NodeTree') + os.sep
        try:
            bpy.ops.wm.append(filepath=os.path.join(directory, stem),
                              directory=directory, filename=stem)
        except RuntimeError:
            continue
        if stem in bpy.data.node_groups:
            loaded.append(stem)
    return loaded


def _override_material(bpy, obj, override):
    """
    Repaint the object's changeable slots, as MuBlE does.

    Only slots whose material name contains 'Changable' are touched. That
    spelling is MuBlE's, and it is not the one its documentation gives
    ('Changeable') -- matching the documented spelling silently repaints
    nothing, which looks like the override being ignored.

    The node group is copied by reference into a fresh material per object, so
    two objects sharing a material type do not clobber each other's colour.
    """
    slots = [i for i, mat in enumerate(obj.data.materials)
             if mat is not None and 'Changable' in mat.name]
    if not slots:
        return None

    material = bpy.data.materials.new('%s.OVERRIDE' % obj.name)
    material.use_nodes = True
    tree = material.node_tree
    output = next((n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL'), None)

    group_name = override.get('node_tree')
    colour = override.get('color')
    if group_name and group_name not in bpy.data.node_groups:
        ## Asked for a shader that never loaded. Falling through to a plain
        ## material would look like a successful override, so say so instead.
        raise LookupError(
            "material '%s' was not appended; check the materials directory"
            % group_name)

    if group_name and output is not None:
        group = tree.nodes.new('ShaderNodeGroup')
        group.node_tree = bpy.data.node_groups[group_name]
        for socket in group.inputs:
            if socket.name == 'Color' and colour:
                socket.default_value = tuple(colour)
        if 'Shader' in group.outputs:
            tree.links.new(group.outputs['Shader'], output.inputs['Surface'])
        if 'Displacement' in group.outputs:
            tree.links.new(group.outputs['Displacement'],
                           output.inputs['Displacement'])
    elif colour:
        ## Colour change with no material change: keep the default shader and
        ## just set its base colour.
        bsdf = tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = tuple(colour)

    for i in slots:
        obj.data.materials[i] = material
    return material


def _cube(bpy, name, position, size, yaw=0.0):
    """A box of the given full extent, centred at `position`."""
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    sx, sy, sz = (s * 0.5 for s in size)
    verts = [(-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz),
             (-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)]
    faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
             (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj.location = tuple(position)
    obj.rotation_euler = (0.0, 0.0, yaw)
    return obj


## MuBlE records a colour word per object. Approximate is fine -- the RGB pass is
## the modality Stage 1 is supposed to learn to discard, and appearance variation
## is the thing being trained against rather than a fidelity target.
COLOURS = {
    'black': (0.02, 0.02, 0.02), 'white': (0.9, 0.9, 0.9),
    'red': (0.6, 0.05, 0.05), 'green': (0.05, 0.4, 0.08),
    'blue': (0.05, 0.1, 0.6), 'yellow': (0.8, 0.7, 0.05),
    'brown': (0.25, 0.14, 0.06), 'grey': (0.4, 0.4, 0.4),
    'gray': (0.4, 0.4, 0.4), 'transparent': (0.8, 0.85, 0.9),
}


def _paint(bpy, obj, spec):
    """
    Give the proxy a material.

    Not cosmetic. On the Blender build robotsim targets, a diffuse surface with
    no material can render black under Cycles while depth and segmentation stay
    correct -- the corruption that looks like success, because the file exists,
    the resolution matches and every pixel is empty.
    """
    material = bpy.data.materials.new('%s.MAT' % obj.name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        rgb = COLOURS.get((spec.get('colour') or '').lower(), (0.5, 0.5, 0.5))
        bsdf.inputs['Base Color'].default_value = rgb + (1.0,)
        if spec.get('material') == 'metal':
            bsdf.inputs['Metallic'].default_value = 0.9
    obj.data.materials.append(material)
    return material


def place_camera(scene, camera):
    """Point a Blender camera where MuBlE's was, if the scene recorded one."""
    params = scene.get('camera')
    if not params:
        return False
    camera.location = tuple(params['position'])
    camera.rotation_euler = tuple(params['rotation'])
    return True


def scenes_in(path):
    """Every handoff file in a directory, in index order."""
    files = [f for f in os.listdir(path) if f.endswith('.json')]
    out = [load(os.path.join(path, f)) for f in sorted(files)]
    return sorted(out, key=lambda s: s.get('index', 0))
