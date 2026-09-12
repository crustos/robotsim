"""
Verify the Blender half of the MuBlE bridge, in Blender.

    blender --background --python tests/muble_blender_test.py

The physics half runs under plain python3 in `muble_test.py`. This half cannot:
appending a .blend, applying a node-group material and reading back a mesh all
need a real Blender. Kept separate rather than merged so the fast tests stay
fast and runnable without Blender installed.

Checks that the RGB pass will actually match MuBlE's render: real meshes rather
than boxes, authored materials preserved, overrides applied only where MuBlE
would apply them, and placement at the mid-bottom origin MuBlE uses.
"""

import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

import muble_bridge                                            # noqa: E402


def muble_root():
    guess = os.path.abspath(os.path.join(HERE, '..', '..', 'MuBlE'))
    marker = os.path.join(guess, 'scene_generation', 'data', 'shapes')
    return guess if os.path.isdir(marker) else None


def clear():
    """Empty the scene, so each check starts from a known world."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def close(a, b, tol=1e-4):
    return abs(a - b) < tol


ROOT = muble_root()
if ROOT is None:
    print('skipping: no MuBlE checkout beside robotsim')
    sys.exit(0)

RAW = os.path.join(ROOT, 'demo_output', 'scene_generaion', 'NS_AP_scenes.json')
if not os.path.isfile(RAW):
    print('skipping: no demo scenes in the MuBlE checkout')
    sys.exit(0)

print('hello muble blender test...')
FAILED = []


def check(name, fn):
    print('%s:' % name)
    try:
        fn()
    except AssertionError as exc:
        FAILED.append(name)
        print('  FAILED: %s' % exc)


def load(index=0):
    import json
    with open(RAW) as handle:
        bundle = json.load(handle)
    return muble_bridge.convert(bundle['scenes'][index], root=ROOT)


# ---------------------------------------------------------------------------

def test_appends_real_meshes():
    """
    Objects come in as their authored geometry, not as boxes.

    Vertex count is the discriminator: the box fallback builds exactly 8
    vertices, so anything above that is a mesh that was actually appended. A
    mug's handle is the difference between a line-art target that looks like a
    mug and one that looks like a crate.
    """
    clear()
    scene = load()
    objects, labels = muble_bridge.to_blender(scene, root=ROOT)

    meshes = [o for o in objects if o.type == 'MESH' and o.name != 'TABLE']
    assert meshes, 'nothing was created'
    counts = {o.name: len(o.data.vertices) for o in meshes}
    boxes = [n for n, c in counts.items() if c <= 8]
    assert not boxes, 'fell back to boxes for: %s' % boxes
    print('  appended real meshes: %s' % counts)


def test_placement_matches_muble():
    """
    Pose is set from `origin`, the mid-bottom, exactly as MuBlE sets it.

    If this used the bounding box centre instead, every object would render half
    its own height above the table -- aligned across all four passes, and wrong
    in all four.
    """
    clear()
    scene = load()
    objects, _labels = muble_bridge.to_blender(scene, root=ROOT)
    by_name = {o.name: o for o in objects}

    for spec in scene['objects']:
        obj = by_name[spec['name']]
        for axis, want in enumerate(spec['origin']):
            assert close(obj.location[axis], want), (spec['name'], axis,
                                                     obj.location[axis], want)
        assert close(obj.scale[0], spec['scale']), (spec['name'], obj.scale[0])
        assert obj.rotation_mode == 'QUATERNION', obj.rotation_mode
        for i, want in enumerate(spec['quaternion']):
            assert close(obj.rotation_quaternion[i], want), (spec['name'], i)
    print('  placement matches MuBlE for %d objects' % len(scene['objects']))


def test_pass_index_is_set():
    """
    Every object carries a distinct index, clear of robotsim's own classes.

    This is the semantic bridge: the integer written here is the label the
    perception network is trained against, and it survives the object-index pass
    exactly rather than being inferred.
    """
    clear()
    scene = load()
    objects, labels = muble_bridge.to_blender(scene, root=ROOT)
    indices = [o.pass_index for o in objects if o.name != 'TABLE']
    assert len(set(indices)) == len(indices), indices
    assert min(indices) > 7, indices
    assert labels, labels
    print('  pass indices %s, labels %s' % (sorted(indices), labels))


def test_authored_materials_survive():
    """
    An object MuBlE does not randomise keeps the appearance it was authored with.

    This is what "visual parity" means in practice: the bridge must not repaint
    a black ceramic mug just because it has a colour field.
    """
    clear()
    scene = load()
    objects, _labels = muble_bridge.to_blender(scene, root=ROOT)
    by_name = {o.name: o for o in objects}

    fixed = [s for s in scene['objects'] if not s.get('material_override')]
    assert fixed, 'no fixed-appearance object in this scene'
    for spec in fixed:
        obj = by_name[spec['name']]
        ## Guard against this passing on the box fallback: a box gets a
        ## generated .MAT and no override either, so without this the test is
        ## satisfied by exactly the outcome it exists to rule out.
        assert len(obj.data.vertices) > 8, (
            '%s fell back to a box; not testing the mesh path' % spec['name'])
        slots = [m for m in obj.data.materials if m is not None]
        assert slots, '%s arrived with no material' % spec['name']
        assert not any('OVERRIDE' in m.name for m in slots), spec['name']
    print('  %d fixed-appearance objects kept their authored materials'
          % len(fixed))


def test_override_applied_where_muble_would():
    """
    A randomised object is repainted, and only in its 'Changable' slots.

    Uses a scene known to contain an override. The spelling of 'Changable' is
    MuBlE's own and differs from its documentation; matching the documented
    spelling repaints nothing at all, which is why this is asserted rather than
    assumed.
    """
    import json
    with open(RAW) as handle:
        bundle = json.load(handle)

    sys.path.insert(0, ROOT)
    import robotsim_export

    target = None
    for i, raw_scene in enumerate(bundle['scenes']):
        exported = robotsim_export.export_scene(raw_scene, i, root=ROOT)
        if any(o.get('material_override') for o in exported['objects']):
            target = exported
            break
    assert target is not None, 'no scene in the demo set has an override'

    clear()
    objects, _labels = muble_bridge.to_blender(target, root=ROOT)
    by_name = {o.name: o for o in objects}

    overridden = [s for s in target['objects'] if s.get('material_override')]
    for spec in overridden:
        obj = by_name[spec['name']]
        names = [m.name for m in obj.data.materials if m is not None]
        assert any('OVERRIDE' in n for n in names), (spec['name'], names)
        ## An override that replaced the slot with a blank material is worse
        ## than no override: it looks applied and renders grey. Assert MuBlE's
        ## own node group is actually wired into the shader.
        want = spec['material_override'].get('node_tree')
        if want:
            material = next(m for m in obj.data.materials
                            if m is not None and 'OVERRIDE' in m.name)
            groups = [n.node_tree.name for n in material.node_tree.nodes
                      if n.type == 'GROUP' and n.node_tree is not None]
            assert want in groups, (spec['name'], want, groups)
        ## Only the changeable slots: an object with a fixed slot alongside a
        ## changeable one must keep the fixed one.
        assert not any('Changable' in n for n in names), (spec['name'], names)
    print('  %d overridden objects repainted: %s'
          % (len(overridden), [s['name'] for s in overridden]))


def test_renders_with_tonal_range():
    """
    The scene actually renders to something other than black.

    robotsim's corpus writer checks this because an unlit render is the
    corruption that looks like success -- the file exists, the resolution
    matches, and every pixel is empty. A bridge that produces geometry nothing
    can see would pass every other check here.
    """
    clear()
    scene = load()
    muble_bridge.to_blender(scene, root=ROOT)

    ## A camera and a light, since the handoff's camera is optional and the
    ## empty factory scene has neither.
    camera_data = bpy.data.cameras.new('CAM')
    camera = bpy.data.objects.new('CAM', camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (0.9, -0.5, 0.6)
    camera.rotation_euler = (1.1, 0.0, 1.0)
    bpy.context.scene.camera = camera

    light_data = bpy.data.lights.new('SUN', type='SUN')
    light_data.energy = 5.0
    light = bpy.data.objects.new('SUN', light_data)
    bpy.context.collection.objects.link(light)
    light.location = (1.0, -1.0, 2.0)

    render = bpy.context.scene.render
    render.resolution_x, render.resolution_y = 96, 72
    render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 4
    ## Not every Blender build ships a denoiser, and Cycles raises rather than
    ## degrading when asked for one it does not have.
    bpy.context.scene.cycles.use_denoising = False
    out = '/tmp/muble_bridge_render.png'
    render.filepath = out
    bpy.ops.render.render(write_still=True)

    image = bpy.data.images.load(out)
    pixels = list(image.pixels)
    rgb = [pixels[i] for i in range(len(pixels)) if i % 4 != 3]
    spread = max(rgb) - min(rgb)
    assert spread > 0.02, 'render has no tonal range (spread %.4f)' % spread
    print('  rendered with tonal range: spread %.3f' % spread)


check('test_appends_real_meshes', test_appends_real_meshes)
check('test_placement_matches_muble', test_placement_matches_muble)
check('test_pass_index_is_set', test_pass_index_is_set)
check('test_authored_materials_survive', test_authored_materials_survive)
check('test_override_applied_where_muble_would', test_override_applied_where_muble_would)
check('test_renders_with_tonal_range', test_renders_with_tonal_range)

print('\n%d failed' % len(FAILED))
sys.exit(1 if FAILED else 0)
