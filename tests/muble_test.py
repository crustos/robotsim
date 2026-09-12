#!/usr/bin/env python3
"""
Tests for the MuJoCo contact backend.

Runs under plain python3, not `../headless.py`, and that is the point: the
backend is free of `bpy` exactly as `drive.py` is, so the physics can be checked
without launching Blender. Every other test in this directory needs a scene;
this one needs a solver.

    ./muble_test.py          # or: make test_muble

Skips cleanly if MuJoCo is not installed, because it is an optional dependency
and a missing optional dependency is not a failure.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

try:
    import muble
    import muble_bridge
except ImportError as exc:                                   # pragma: no cover
    print('skipping muble tests: %s' % exc)
    sys.exit(0)

if not muble.HAVE_MUJOCO:                                    # pragma: no cover
    print('skipping muble tests: MuJoCo not installed (pip install mujoco)')
    sys.exit(0)

print('hello muble test...')

DT = 1.0 / 60.0
WHEELS = [(-0.25, 0.3), (0.25, 0.3), (-0.25, -0.3), (0.25, -0.3)]


class FakeDrive:
    """The seam only ever reads `drive` for levelling, which this backend does itself."""


def close(a, b, tol=1e-2):
    return abs(a - b) < tol


def make(mu=1.0, mass=10.0, **kw):
    contact = muble.MujocoContact(mass=mass, mu=mu, contact_points=WHEELS, **kw)
    contact.add_ground()
    return contact


def run(contact, v_cmd=0.0, omega_cmd=0.0, ticks=120, dt=DT):
    """Drive for a while, returning the final pose, twist and contact record."""
    contact.settle()
    pose = contact.pose
    drive = FakeDrive()
    v, omega, info = 0.0, 0.0, None
    for _ in range(ticks):
        pose, (v, omega), info = contact.resolve(
            drive, pose, pose, (v_cmd, omega_cmd), dt)
    return pose, (v, omega), info


# ---------------------------------------------------------------------------

def test_rests_on_ground():
    """Gravity settles the robot onto its wheels rather than through them."""
    contact = make().build()
    contact.set_pose(0.0, 0.0, 3.0, 0.0)
    info = contact.settle(seconds=1.5)
    _x, _y, z, _yaw = contact.pose
    ## Hull centre sits a wheel radius plus half its own height above the floor.
    assert close(z, 0.25, 0.02), z
    assert info.grounded, info
    assert not info.airborne, info
    ## The whole weight is carried, not some fraction: a robot resting on two of
    ## its four wheels is a geometry bug that otherwise passes every other test.
    assert close(info.wheel_load, 10.0 * 9.81, 1.0), info.wheel_load
    print('  rests on ground: z=%.3f load=%.1fN' % (z, info.wheel_load))


def test_reaches_commanded_speed():
    """With grip to spare the dynamic backend tracks the command."""
    contact = make(mu=1.0).build()
    _pose, (v, _omega), info = run(contact, v_cmd=1.5)
    assert close(v, 1.5, 0.05), v
    ## Well inside the friction circle: this surface is not the limit.
    assert info.traction < 0.3, info.traction
    assert not info.slid, info
    print('  tracks command: v=%.3f traction=%.2f' % (v, info.traction))


def test_slip_has_consequences():
    """
    The headline difference from the kinematic backend.

    RayContact reports slip as a number and then moves the robot the commanded
    distance anyway. Here the tyres saturate, the robot falls short, and the
    shortfall *is* the slip.
    """
    grippy = make(mu=1.0).build()
    (_x, gy, _z, _yaw), (gv, _o), ginfo = run(grippy, v_cmd=1.5)

    icy = make(mu=0.05).build()
    (_x, iy, _z, _yaw), (iv, _o), iinfo = run(icy, v_cmd=1.5)

    assert iy < gy * 0.5, (iy, gy)
    assert iv < gv * 0.8, (iv, gv)
    assert iinfo.traction >= 0.99, iinfo.traction
    assert iinfo.slid, iinfo
    assert not ginfo.slid, ginfo
    print('  slip bites: tarmac y=%.2f v=%.2f | ice y=%.2f v=%.2f' %
          (gy, gv, iy, iv))


def test_acceleration_is_grip_limited():
    """Peak acceleration is mu*g, whatever the motors are asked for."""
    contact = make(mu=0.3).build()
    contact.settle()
    pose = contact.pose
    drive = FakeDrive()
    v = 0.0
    ## One tick from rest at an impossible command: the tyres, not the command,
    ## decide what happens.
    pose, (v, _omega), _info = contact.resolve(drive, pose, pose, (50.0, 0.0), DT)
    limit = 0.3 * 9.81 * DT
    assert v <= limit * 1.3, (v, limit)
    assert v > limit * 0.5, (v, limit)
    print('  grip-limited accel: v=%.4f after one tick, mu*g*dt=%.4f' % (v, limit))


def test_stops_at_wall():
    """A hull contact blocks the robot and takes its speed with it."""
    contact = make(mu=1.0)
    contact.add_box('WALL', (0.0, 3.0, 0.5), (4.0, 0.2, 1.0))
    contact.build()
    (_x, y, _z, _yaw), (v, _omega), info = run(contact, v_cmd=1.5)
    ## Wall face at y=2.9, hull half-length 0.4, so the robot stops at y=2.5.
    assert close(y, 2.5, 0.1), y
    assert close(v, 0.0, 0.05), v
    assert info.blocked, info
    assert info.geom == 'WALL', info.geom
    assert info.force > 0.0, info.force
    print('  blocked by wall: y=%.3f v=%.3f F=%.1fN' % (y, v, info.force))


def test_momentum_carries_through_release():
    """Letting off the throttle coasts rather than stopping dead."""
    contact = make(mu=1.0).build()
    contact.settle()
    pose = contact.pose
    drive = FakeDrive()
    v = 0.0
    for _ in range(90):
        pose, (v, _o), _i = contact.resolve(drive, pose, pose, (2.0, 0.0), DT)
    moving = v
    ## Command zero and step once: a kinematic model would be stopped already.
    pose, (v, _o), _i = contact.resolve(drive, pose, pose, (0.0, 0.0), DT)
    assert moving > 1.5, moving
    assert v > moving * 0.5, (v, moving)
    print('  momentum: %.3f m/s at release, %.3f m/s one tick later' % (moving, v))


def test_turns_at_commanded_rate():
    """Differential drive produces yaw from unequal wheel forces."""
    contact = make(mu=1.0).build()
    _pose, (_v, omega), _info = run(contact, v_cmd=0.0, omega_cmd=1.0)
    assert close(omega, 1.0, 0.1), omega
    print('  spins on the spot: omega=%.3f' % omega)


def test_arc_radius():
    """A twist of (v, omega) traces a circle of radius v/omega."""
    contact = make(mu=1.0).build()
    contact.settle()
    pose = contact.pose
    drive = FakeDrive()
    v = 0.0
    ## Skip the spin-up transient before measuring, or the ramp to commanded
    ## yaw rate is scored as a radius error.
    for _ in range(60):
        pose, (v, _o), _i = contact.resolve(drive, pose, pose, (1.0, 1.0), DT)
    x0, y0, _z, yaw0 = pose
    for _ in range(60):
        pose, (v, _o), _i = contact.resolve(drive, pose, pose, (1.0, 1.0), DT)
    x1, y1, _z, yaw1 = pose
    turned = abs(yaw1 - yaw0)
    chord = math.hypot(x1 - x0, y1 - y0)
    ## chord = 2 r sin(theta/2)
    radius = chord / (2.0 * math.sin(turned * 0.5))
    assert close(radius, 1.0, 0.15), radius
    print('  arc radius: %.3f m (commanded v/omega = 1.000)' % radius)


def test_climbs_ramp():
    """A slope is driven up, and the robot ends up higher than it started."""
    contact = make(mu=1.0)
    ## A shallow wedge: a box rotated about its long axis is not expressible
    ## through add_box's yaw-only orientation, so use a mesh.
    verts = [(-2, 0, 0), (2, 0, 0), (-2, 4, 0), (2, 4, 0),
             (-2, 4, 0.8), (2, 4, 0.8)]
    faces = [(0, 1, 3), (0, 3, 2), (2, 3, 5), (2, 5, 4),
             (0, 2, 4), (0, 4, 1), (1, 4, 5), (1, 5, 3)]
    contact.add_mesh('RAMP', verts, faces, pos=(0.0, 1.0, 0.0))
    contact.build()
    (_x, y, z, _yaw), (_v, _o), info = run(contact, v_cmd=1.2, ticks=180)
    assert z > 0.4, z
    assert y > 2.0, y
    assert info.grounded, info
    print('  climbed ramp: y=%.2f z=%.3f' % (y, z))


def test_teleport_is_respected():
    """
    Moving the robot from outside the contact model is honoured, not overwritten.

    The solver holds its own state between steps, which is what carries momentum.
    A caller that repositions the robot -- a reset, a scripted pose, an episode
    boundary -- has to win, and the model detects that by comparing against the
    pose it last handed back rather than by being told.
    """
    contact = make(mu=1.0).build()
    contact.settle()
    pose = contact.pose
    drive = FakeDrive()
    v = 0.0
    for _ in range(30):
        pose, (v, _o), _i = contact.resolve(drive, pose, pose, (1.5, 0.0), DT)
    moved = (pose[0], 20.0, pose[2], pose[3])        ## caller teleports it
    pose, (v, _o), _i = contact.resolve(drive, moved, moved, (1.5, 0.0), DT)
    assert close(pose[1], 20.0, 0.1), pose
    print('  teleport respected: y=%.3f' % pose[1])


def test_matches_kinematic_when_unlimited():
    """
    With plenty of grip the two backends agree, which is what makes this a
    drop-in rather than a different simulator.

    Not compared against RayContact directly -- that needs Blender -- but against
    the kinematic prediction RayContact would make on open ground: commanded
    speed, integrated.
    """
    contact = make(mu=2.0).build()
    ticks = 180
    (_x, y, _z, _yaw), (v, _o), _info = run(contact, v_cmd=1.0, ticks=ticks)
    predicted = 1.0 * ticks * DT
    ## Short by the spin-up distance only.
    assert y > predicted - 0.15, (y, predicted)
    assert y <= predicted + 0.01, (y, predicted)
    print('  agrees with kinematic: y=%.3f vs predicted %.3f' % (y, predicted))


def test_interface_shape():
    """resolve() returns what drive.DriveBase.step expects, and nothing else."""
    contact = make().build()
    contact.settle()
    result = contact.resolve(FakeDrive(), contact.pose, contact.pose, (1.0, 0.0), DT)
    assert len(result) == 3, result
    pose, velocity, info = result
    assert len(pose) == 4, pose
    assert len(velocity) == 2, velocity
    ## Duck-typed against contact.Contact so existing callers do not care which
    ## backend produced the record.
    for field in ('blocked', 'object', 'normal', 'distance', 'grounded',
                  'ground_z', 'slid'):
        assert hasattr(info, field), field
    assert isinstance(contact, muble.ContactModel), 'must satisfy the seam'
    print('  interface shape ok: %r' % (info,))


def test_zero_dt_is_a_noop():
    contact = make().build()
    contact.settle()
    pose = contact.pose
    out, vel, info = contact.resolve(FakeDrive(), pose, pose, (1.0, 0.0), 0.0)
    assert out == pose, (out, pose)
    assert info is None
    print('  zero dt is a no-op')


# -- the MuBlE scene bridge -------------------------------------------------

def handoff():
    """A minimal handoff, shaped exactly as robotsim_export.py writes one."""
    return {
        'format': muble_bridge.FORMAT, 'version': 1, 'index': 0,
        ## Big enough to hold everything else in the fixture. A table that
        ## stops short of the obstacle makes the robot drive off the edge, and
        ## the resulting failure reads as "physics broken" rather than
        ## "test geometry inconsistent".
        'table': {'name': 'TABLE', 'id': 0,
                  'position': [0.0, 0.0, -0.025], 'size': [10.0, 10.0, 0.05]},
        'objects': [
            {'id': 1, 'name': 'mug_00', 'label': 'mug',
             'position': [0.0, 3.0, 0.5], 'size': [4.0, 0.2, 1.0],
             'yaw': 0.0, 'colour': 'black', 'material': 'ceramic'},
            {'id': 2, 'name': 'mug_01', 'label': 'mug',
             'position': [-3.0, 0.0, 0.04], 'size': [0.1, 0.1, 0.08],
             'yaw': 0.5, 'colour': 'white', 'material': 'ceramic'},
        ],
    }


def test_bridge_labels_are_disjoint_from_robotsim():
    """
    Imported objects must not land on robotsim's own semantic indices.

    Sharing an index with 'ground' or 'obstacle' raises nothing and trains the
    segmentation head to call a mug a floor, so it is worth an assertion.
    """
    scene = handoff()
    table = muble_bridge.labels(scene)
    ## robotsim's SEGMENT_CLASSES occupy 0-5, procedural obstacles use 7.
    assert min(table.values()) > 7, table
    ## Two mugs, one class: a semantic map groups by label, not by instance.
    assert len(table) == 2, table
    assert 'mug' in table and 'table' in table, table
    ## Distinct objects keep distinct pass indices even so.
    ids = {muble_bridge.label_of(o) for o in scene['objects']}
    assert len(ids) == 2, ids
    print('  labels disjoint and grouped: %s' % table)


def test_bridge_builds_physics():
    """A MuBlE scene becomes a world a robot can be driven around."""
    scene = handoff()
    contact = muble_bridge.build_contact(
        scene, mu=1.0, mass=10.0, contact_points=WHEELS)
    names = [s['name'] for s in contact.statics]
    assert 'TABLE' in names and 'mug_00' in names, names
    (_x, y, _z, _yaw), (v, _o), info = run(contact, v_cmd=1.5)
    ## 'mug_00' here is a wall-sized box at y=3; the robot should stop against it.
    assert info.blocked, info
    assert info.geom == 'mug_00', info.geom
    assert close(y, 2.5, 0.15), y
    print('  drove a MuBlE scene: stopped at y=%.2f on %s' % (y, info.geom))


def test_bridge_reads_raw_muble_scenes():
    """
    A raw MuBlE bundle is accepted, not just an exported handoff.

    Users point the bridge at MuBlE's own output first; refusing it would be
    correct and unhelpful.
    """
    import json
    import tempfile
    raw = {'scenes': [{
        'image_index': 4,
        'objects': [{
            'file': 'mug2', 'name': 'mug', 'colour': 'black',
            ## MuBlE puts the origin at the object's mid-bottom.
            '3d_coords': [0.5, 0.1, 0.0],
            'bbox': {'x': 0.08, 'y': 0.06, 'z': 0.072},
            'orientation': [0.7996, 0.0, 0.0, 0.6005],
            'weight_gt': 98.4, 'movability': 'portable',
        }],
    }]}
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as fh:
        json.dump(raw, fh)
        path = fh.name
    try:
        scene = muble_bridge.load(path)
    finally:
        os.unlink(path)

    assert scene['index'] == 4, scene['index']
    obj = scene['objects'][0]
    ## Mid-bottom to centre: z must be lifted by half the height, or every
    ## object sits half-buried in the table.
    assert close(obj['position'][2], 0.036, 1e-3), obj['position']
    ## MuBlE stores the same rotation twice, as a quaternion and as 73.8 degrees.
    assert close(obj['yaw'], math.radians(73.81), 1e-3), obj['yaw']
    assert close(obj['mass'], 0.0984, 1e-4), obj['mass']
    print('  raw MuBlE scene read: yaw=%.3f rad, z=%.3f' %
          (obj['yaw'], obj['position'][2]))


def test_origin_and_position_are_not_interchangeable():
    """
    The mid-bottom / centre distinction, asserted so it cannot regress.

    MuBlE's `3d_coords` is an object's mid-bottom and is where its real mesh
    goes. The bounding box centre sits half an object higher and belongs to the
    box fallback. Swapping them buries every object half its own height in the
    table or floats it, and both look like a physics bug rather than a
    bookkeeping one.
    """
    scene = handoff_with_assets() or handoff()
    for obj in scene['objects']:
        if 'origin' not in obj:
            continue
        lift = obj['position'][2] - obj['origin'][2]
        assert close(lift, obj['size'][2] * 0.5, 1e-6), (obj['name'], lift)
    print('  origin is mid-bottom, position is centre: distinct by height/2')


def muble_root():
    """A sibling MuBlE checkout, or None."""
    here = os.path.dirname(os.path.abspath(__file__))
    guess = os.path.join(here, '..', '..', 'MuBlE')
    guess = os.path.abspath(guess)
    marker = os.path.join(guess, 'scene_generation', 'data', 'shapes')
    return guess if os.path.isdir(marker) else None


def handoff_with_assets():
    """A real exported scene, if a MuBlE checkout is next door."""
    root = muble_root()
    if root is None:
        return None
    raw = os.path.join(root, 'demo_output', 'scene_generaion',
                       'NS_AP_scenes.json')
    if not os.path.isfile(raw):
        return None
    return muble_bridge.load(raw, root=root)


def test_physics_uses_shipped_hulls():
    """
    Collision geometry is MuBlE's convex decomposition, not a bounding box.

    A mug's handle and its hollow are the difference between a robot that can
    be blocked by a scene and one that is blocked by the boxes around it.
    """
    scene = handoff_with_assets()
    if scene is None:
        print('  skipped: no MuBlE checkout beside robotsim')
        return

    meshed = [o for o in scene['objects'] if o.get('collision')]
    assert meshed, 'no object resolved any collision hulls'

    contact = muble_bridge.build_contact(
        scene, mu=1.0, mass=2.0, size=(0.1, 0.12, 0.06), wheel_radius=0.02,
        contact_points=[(-0.04, 0.05), (0.04, 0.05),
                        (-0.04, -0.05), (0.04, -0.05)])
    kinds = {}
    for static in contact.statics:
        kinds[static['kind']] = kinds.get(static['kind'], 0) + 1
    assert kinds.get('file', 0) > kinds.get('box', 0), kinds

    ## The scene must still be something a robot can stand on and drive in.
    contact.settle()
    info, _loads = contact.read_contacts()
    assert info.grounded, info
    print('  physics on real hulls: %d hull geoms, %d boxes'
          % (kinds.get('file', 0), kinds.get('box', 0)))


def test_hulls_fall_back_per_object():
    """
    A missing asset costs one approximate object, not the whole scene.

    Per-object fallback rather than per-scene: a partial checkout should still
    produce a usable world.
    """
    scene = handoff()
    scene['objects'][0]['collision'] = ['meshes/train/nope/nope_hull_1.stl']
    scene['assets'] = {'root': '/nonexistent', 'meshes': 'meshes/train'}
    contact = muble_bridge.build_contact(
        scene, mu=1.0, mass=10.0, contact_points=WHEELS)
    names = [s['name'] for s in contact.statics]
    ## Unresolvable hull -> box named for the object, rather than an exception.
    assert 'mug_00' in names, names
    assert all(s['kind'] != 'file' for s in contact.statics), names
    print('  missing asset falls back to a box: %d statics' % len(names))


def test_exporter_and_bridge_agree_on_assets():
    """
    The two repositories describe the same scene the same way.

    They cannot import each other, so a little logic is duplicated in both. This
    is the assertion that keeps the duplicate honest.
    """
    root = muble_root()
    if root is None:
        print('  skipped: no MuBlE checkout beside robotsim')
        return
    sys.path.insert(0, root)
    try:
        import robotsim_export
    except ImportError:
        print('  skipped: robotsim_export not importable')
        return

    raw_path = os.path.join(root, 'demo_output', 'scene_generaion',
                            'NS_AP_scenes.json')
    if not os.path.isfile(raw_path):
        print('  skipped: no demo scenes')
        return

    with open(raw_path) as handle:
        bundle = json.load(handle)
    exported = robotsim_export.export_scene(bundle['scenes'][0], 0, root=root)
    converted = muble_bridge.convert(bundle['scenes'][0], root=root)

    for a, b in zip(exported['objects'], converted['objects']):
        for field in ('shape', 'origin', 'position', 'scale', 'quaternion',
                      'collision'):
            assert a[field] == b[field], (field, a[field], b[field])
    print('  exporter and bridge agree on %d objects'
          % len(exported['objects']))


TESTS = [test_rests_on_ground, test_reaches_commanded_speed,
         test_slip_has_consequences, test_acceleration_is_grip_limited,
         test_stops_at_wall, test_momentum_carries_through_release,
         test_turns_at_commanded_rate, test_arc_radius, test_climbs_ramp,
         test_teleport_is_respected, test_matches_kinematic_when_unlimited,
         test_interface_shape, test_zero_dt_is_a_noop,
         test_bridge_labels_are_disjoint_from_robotsim,
         test_bridge_builds_physics, test_bridge_reads_raw_muble_scenes,
         test_origin_and_position_are_not_interchangeable,
         test_physics_uses_shipped_hulls, test_hulls_fall_back_per_object,
         test_exporter_and_bridge_agree_on_assets]


if __name__ == '__main__':
    failed = 0
    for test in TESTS:
        print('%s:' % test.__name__)
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print('  FAILED: %s' % exc)
    print('\n%d/%d passed' % (len(TESTS) - failed, len(TESTS)))
    sys.exit(1 if failed else 0)
