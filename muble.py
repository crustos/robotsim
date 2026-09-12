"""
Dynamic contact via MuJoCo -- the optional backend the kinematic one is not.

`contact.RayContact` answers "where does the robot end up" by casting rays and
moving the pose. It is fast, it has no mass, and slip is a number it reports
rather than a thing that happens. This module answers the same question by
integrating rigid-body dynamics, so the robot has weight, momentum carries
through a collision, and a wheel asked for more acceleration than the ground
can give simply spins.

It plugs into `drive.ContactModel`, the same seam `RayContact` uses -- the drive
models command a velocity and ask what pose they actually get, and neither side
has to know which kind of answer is on the other. Nothing else in robotsim
changes, and a robot built without this module behaves exactly as it did.

Named for MuBlE (arXiv:2503.02834), which couples MuJoCo physics to Blender
rendering for manipulation. robotsim makes the opposite trade by default --
kinematic contact, effort spent on the render modalities and on executing real
firmware -- so the two are complementary rather than competing. This is the
join: MuBlE's physics under robotsim's cameras. `muble_bridge.py` imports MuBlE
scenes directly when it is installed; this module needs only `mujoco` and works
without it.

Deliberately free of `bpy`, exactly as `drive.py` is, for two reasons: the
physics can be tested without launching Blender, and the same backend can drive
a headless training loop. `from_blender()` is the only part that touches a
scene, and it is imported lazily.

    import muble
    contact = muble.MujocoContact(mass=12.0, contact_points=[...])
    contact.add_ground()
    contact.add_box('WALL', (2, 0, 0.5), (0.2, 4, 1))
    contact.build()
    robot.drive.contact = contact

Cost is one `mj_step` per substep rather than a handful of ray casts, so this is
the accurate option, not the cheap one.
"""

import math
import os

try:
    import mujoco
    import numpy as np
    HAVE_MUJOCO = True
except ImportError:                                  # pragma: no cover
    mujoco = None
    np = None
    HAVE_MUJOCO = False

from drive import ContactModel


## Blender's +Z is up and robotsim measures yaw about +Z from +Y, so forward in
## the body frame is +Y and right is +X. MuJoCo is also Z-up, which means the
## body frame maps across as a plain rotation about Z with no axis permutation.
GRAVITY = -9.81


def yaw_to_quat(yaw):
    """Yaw about +Z as a MuJoCo (w, x, y, z) quaternion."""
    return (math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5))


def quat_to_yaw(quat):
    """Heading of a (w, x, y, z) quaternion, as rotation about +Z from +Y."""
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def forward_axis(yaw):
    """World unit vector the robot is facing: body +Y rotated by yaw."""
    return (-math.sin(yaw), math.cos(yaw), 0.0)


def right_axis(yaw):
    """World unit vector out of the robot's right side: body +X rotated by yaw."""
    return (math.cos(yaw), math.sin(yaw), 0.0)


class Contact:
    """
    What happened during one resolve().

    Field-for-field compatible with `contact.Contact`, so callers and tests that
    already read `blocked`, `grounded`, `slid` and friends do not care which
    backend produced it. `geom` and `force` are additional, because a dynamics
    engine knows things a ray cast cannot: how hard the impact was, and how much
    of the available grip the tyres are using.
    """

    def __init__(self, blocked=False, obj=None, normal=None, distance=None,
                 grounded=False, ground_z=None, slid=False, geom=None,
                 force=0.0, wheel_load=0.0, traction=0.0, airborne=False):
        self.blocked = blocked
        self.object = obj
        self.normal = normal
        self.distance = distance
        self.grounded = grounded
        self.ground_z = ground_z
        self.slid = slid
        ## Name of the geom that blocked, when there is no Blender object to
        ## point at -- which is the normal case in a headless run.
        self.geom = geom
        ## Peak normal force of the blocking contact, in newtons.
        self.force = force
        ## Total normal load carried by the wheels. Zero means airborne.
        self.wheel_load = wheel_load
        ## Fraction of available grip in use, across all wheels. At 1.0 the
        ## tyres are saturated and the robot is sliding rather than driving.
        self.traction = traction
        self.airborne = airborne

    def __repr__(self):
        return '<Contact%s%s%s%s%s>' % (
            ' blocked' if self.blocked else '',
            ' slid' if self.slid else '',
            ' grounded' if self.grounded else '',
            ' airborne' if self.airborne else '',
            (' on=%s' % self.name) if self.name else '')

    @property
    def name(self):
        """Whatever we can call the thing that was hit, object or geom."""
        if self.object is not None:
            return getattr(self.object, 'name', str(self.object))
        return self.geom


class MujocoContact(ContactModel):
    """
    Rigid-body contact: the robot as a mass with wheels, in a MuJoCo world.

    The base is a single free body carrying a box geom for its hull and a sphere
    at each contact point for its wheels. Gravity, collision and momentum come
    from the solver. Traction does not -- it comes from the tyre model in
    `apply_tyres`, because a sphere sliding on a plane is not a wheel and a
    solver told to treat it as one produces a robot that either cannot turn or
    cannot stop.

    The wheel geoms are therefore given near-zero friction and every tyre force
    is applied explicitly, which is how vehicle simulation is normally done and
    is what makes the friction circle -- and so wheelspin -- available at all.

    `mu` is the tyre friction coefficient, the single knob that decides whether
    the robot is on tarmac (1.0) or ice (0.05).

    One MuJoCo detail forces the design and is worth stating plainly: contact
    friction between two geoms is combined by taking the *maximum* of the two,
    not the minimum or the product. A "frictionless" wheel on a high-friction
    floor is therefore not frictionless at all -- the floor wins -- and the
    solver's own tangential force ends up fighting the tyre model almost exactly,
    producing a robot that is grounded, has full load, reports saturated grip and
    still crawls. So every geom here is built with `solver_mu`, low enough that
    the solver contributes normal force and essentially nothing tangential, and
    all traction is applied explicitly. `mu` is then genuinely the only friction
    in the model rather than one of two competing sources.
    """

    def __init__(self, mass=10.0, size=(0.6, 0.8, 0.3), contact_points=None,
                 wheel_radius=0.1, ride_height=None, mu=1.0, mu_hull=0.4,
                 timestep=0.002, gravity=GRAVITY, damping=0.02,
                 solver_mu=0.01, solref=(0.02, 1.0), max_substeps=200,
                 stiction=1e-3, blocked_threshold=0.25, tyre_tau=0.05,
                 scene=None):
        if not HAVE_MUJOCO:
            raise ImportError(
                'muble.MujocoContact needs MuJoCo: pip install mujoco. '
                'Use contact.RayContact for the kinematic backend instead.')
        self.mass = mass
        self.size = size
        self.wheel_radius = wheel_radius
        ## Ride height is the hull's underside clearance. Defaulting it to the
        ## wheel radius is what puts the hull exactly on top of its wheels, so
        ## the body sits at the height the wheels hold it at rather than at an
        ## arbitrary offset that would leave it jammed into the ground.
        self.ride_height = ride_height if ride_height is not None else wheel_radius
        self.contact_points = list(contact_points or [(-0.3, 0.3), (0.3, 0.3),
                                                      (-0.3, -0.3), (0.3, -0.3)])
        self.mu = mu
        self.mu_hull = mu_hull
        self.timestep = timestep
        self.gravity = gravity
        self.damping = damping
        ## Solver-side contact friction, deliberately near zero. See the class
        ## docstring: this exists so the solver does not silently supply a
        ## second, competing source of traction.
        self.solver_mu = solver_mu
        self.solref = solref
        ## Guard against a caller passing a huge dt and asking the solver to run
        ## for minutes inside one step().
        self.max_substeps = max_substeps
        self.stiction = stiction
        ## Tyre relaxation time: how quickly a wheel tries to erase its own
        ## slip. Deliberately not `dt` -- a force sized to cancel slip within
        ## one tick is enormous, saturates the friction circle on every step
        ## and leaves the robot permanently at its grip limit, which reads as
        ## "no traction anywhere" no matter how good the surface is.
        self.tyre_tau = tyre_tau
        ## Normal force above which a non-wheel contact counts as "blocked"
        ## rather than a graze, in newtons.
        self.blocked_threshold = blocked_threshold
        self.scene = scene

        self.statics = []
        self.model = None
        self.data = None
        self.body_id = None
        self.geom_objects = {}       ## geom name -> Blender object, when known
        self.wheel_geoms = set()
        self.hull_geoms = set()
        ## Last pose we handed back, so an external teleport can be told apart
        ## from the solver's own motion. Without this, writing the pose back
        ## into MuJoCo every step would destroy the velocity state that *is*
        ## the momentum we are here to model.
        self.last_pose = None

    # -- world building -----------------------------------------------------

    def add_ground(self, z=0.0, mu=None, name='GROUND'):
        """An infinite ground plane at `z`."""
        self.statics.append({
            'kind': 'plane', 'name': name, 'pos': (0.0, 0.0, z),
            'size': (0.0, 0.0, 1.0), 'quat': (1.0, 0.0, 0.0, 0.0),
            'mu': self.solver_mu if mu is None else mu, 'obj': None})
        return self

    def add_box(self, name, pos, size, yaw=0.0, mu=None, obj=None):
        """
        A static box. `size` is the full extent, as Blender reports dimensions,
        and is halved on the way in because MuJoCo takes half-extents -- the
        single most common way to build a world that is twice the size intended.
        """
        self.statics.append({
            'kind': 'box', 'name': name, 'pos': tuple(pos),
            'size': tuple(s * 0.5 for s in size), 'quat': yaw_to_quat(yaw),
            'mu': self.solver_mu if mu is None else mu, 'obj': obj})
        return self

    def add_mesh(self, name, vertices, faces, pos=(0, 0, 0), yaw=0.0,
                 mu=None, obj=None):
        """
        A static triangle mesh, for terrain and anything not box-shaped.

        Faces must be triangles; callers holding quads should triangulate first.
        MuJoCo treats a mesh geom as a convex hull for collision unless it is
        supplied as an explicit set of triangles, which is why terrain is worth
        passing through `add_heightfield` instead where the shape is a surface.
        """
        self.statics.append({
            'kind': 'mesh', 'name': name, 'pos': tuple(pos),
            'quat': yaw_to_quat(yaw), 'mu': self.solver_mu if mu is None else mu,
            'vertices': [tuple(v) for v in vertices],
            'faces': [tuple(f) for f in faces], 'obj': obj})
        return self

    def add_mesh_file(self, name, path, pos=(0, 0, 0), quat=None, scale=1.0,
                      yaw=None, mu=None, obj=None):
        """
        A static geom loaded straight from a mesh file on disk (STL, OBJ, MSH).

        MuJoCo parses the file itself, so nothing here reads vertices. That
        matters for MuBlE's convex-hull decompositions: each object ships as a
        stack of hull STLs, and a mesh geom in MuJoCo is convex anyway, so the
        decomposition maps onto one geom per hull with no conversion at all.

        `scale` is uniform and applied to the mesh asset rather than the geom,
        which is why each instance gets its own asset entry -- two objects using
        the same file at different scales are two different meshes as far as the
        compiler is concerned.

        Orientation is a full (w, x, y, z) quaternion rather than a yaw, because
        an object resting on its side is a pose a single angle cannot express.
        `yaw` stays available for the flat case.
        """
        if quat is None:
            quat = yaw_to_quat(yaw or 0.0)
        self.statics.append({
            'kind': 'file', 'name': name, 'pos': tuple(pos),
            'quat': tuple(quat), 'path': os.path.abspath(path),
            'scale': scale, 'mu': self.solver_mu if mu is None else mu,
            'obj': obj})
        return self

    def mjcf(self):
        """The world as MJCF XML. Exposed because a physics bug is read, not guessed."""
        out = ['<mujoco model="robotsim">',
               '  <compiler angle="radian"/>',
               '  <option timestep="%g" gravity="0 0 %g" integrator="implicitfast"/>'
               % (self.timestep, self.gravity),
               '  <default>',
               '    <geom solref="%g %g"/>' % self.solref,
               '  </default>']

        meshes = [s for s in self.statics if s['kind'] == 'mesh']
        files = [s for s in self.statics if s['kind'] == 'file']
        if meshes or files:
            out.append('  <asset>')
            for s in meshes:
                verts = ' '.join('%g %g %g' % v for v in s['vertices'])
                faces = ' '.join('%d %d %d' % f for f in s['faces'])
                out.append('    <mesh name="%s_mesh" vertex="%s" face="%s"/>'
                           % (s['name'], verts, faces))
            for s in files:
                out.append('    <mesh name="%s_mesh" file="%s" scale="%g %g %g"/>'
                           % (s['name'], s['path'],
                              s['scale'], s['scale'], s['scale']))
            out.append('  </asset>')

        out.append('  <worldbody>')
        for s in self.statics:
            if s['kind'] == 'plane':
                out.append('    <geom name="%s" type="plane" pos="%g %g %g" '
                           'size="0 0 1" friction="%g 0.005 0.0001"/>'
                           % ((s['name'],) + s['pos'] + (s['mu'],)))
            elif s['kind'] == 'box':
                out.append('    <geom name="%s" type="box" pos="%g %g %g" '
                           'quat="%g %g %g %g" size="%g %g %g" '
                           'friction="%g 0.005 0.0001"/>'
                           % ((s['name'],) + s['pos'] + s['quat'] + s['size']
                              + (s['mu'],)))
            else:
                ## Both the inline-vertex and on-disk cases land here: they
                ## differ in how the asset was declared, not in how it is placed.
                out.append('    <geom name="%s" type="mesh" mesh="%s_mesh" '
                           'pos="%g %g %g" quat="%g %g %g %g" '
                           'friction="%g 0.005 0.0001"/>'
                           % ((s['name'], s['name']) + s['pos'] + s['quat']
                              + (s['mu'],)))

        ## The hull sits a wheel-radius up so its underside clears the ground,
        ## and the wheels hang below it at exactly that radius. Getting this
        ## wrong is how a robot ends up resting on its belly with the wheels
        ## buried, which looks like "it will not move" rather than a geometry bug.
        hull_z = self.ride_height + self.size[2] * 0.5
        out.append('    <body name="base" pos="0 0 %g">' % hull_z)
        out.append('      <freejoint name="base"/>')
        out.append('      <geom name="hull" type="box" size="%g %g %g" '
                   'mass="%g" friction="%g 0.005 0.0001"/>'
                   % (self.size[0] * 0.5, self.size[1] * 0.5,
                      self.size[2] * 0.5, self.mass, self.solver_mu))
        for i, (ox, oy) in enumerate(self.contact_points):
            ## Wheel centre, relative to the hull's own centre.
            wz = -self.size[2] * 0.5 - (self.ride_height - self.wheel_radius)
            out.append('      <geom name="wheel%d" type="sphere" pos="%g %g %g" '
                       'size="%g" mass="0.001" friction="%g 0.005 0.0001"/>'
                       % (i, ox, oy, wz, self.wheel_radius, self.solver_mu))
        out.append('    </body>')
        out.append('  </worldbody>')
        out.append('</mujoco>')
        return '\n'.join(out)

    def build(self):
        """Compile the world. Call once, after the statics are in."""
        self.model = mujoco.MjModel.from_xml_string(self.mjcf())
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base')
        self.wheel_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'wheel%d' % i)
            for i in range(len(self.contact_points))]
        self.wheel_geoms = set(self.wheel_ids)
        self.hull_geoms = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'hull')}
        for s in self.statics:
            if s.get('obj') is not None:
                self.geom_objects[s['name']] = s['obj']
        mujoco.mj_forward(self.model, self.data)
        self.last_pose = None
        return self

    def ensure_built(self):
        if self.model is None:
            self.build()

    # -- state --------------------------------------------------------------

    @property
    def pose(self):
        """Current (x, y, z, yaw) of the base, as the seam means it."""
        q = self.data.qpos
        return (float(q[0]), float(q[1]), float(q[2]), quat_to_yaw(q[3:7]))

    def set_pose(self, x, y, z, yaw, keep_velocity=False):
        """Place the base, optionally preserving its momentum."""
        self.ensure_built()
        self.data.qpos[0:3] = (x, y, z)
        self.data.qpos[3:7] = yaw_to_quat(yaw)
        if not keep_velocity:
            self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.last_pose = self.pose

    def snap(self, root):
        """
        Drop the robot onto whatever is beneath it, right now.

        Mirrors `RayContact.snap` so `enable_contact()` can call either. Settling
        under gravity rather than ray-casting a height means the robot also comes
        to rest *level*, which matters on a slope where a single probe would have
        left it intersecting the ground.
        """
        self.ensure_built()
        loc = root.location
        self.set_pose(loc.x, loc.y, loc.z, getattr(root.rotation_euler, 'z', 0.0))
        info = self.settle()
        x, y, z, _yaw = self.pose
        loc.z = z
        return info

    def settle(self, seconds=0.5):
        """Let the world run with no drive input until the robot rests."""
        self.ensure_built()
        steps = max(1, int(seconds / self.timestep))
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.last_pose = self.pose
        return self.read_contacts()[0]

    # -- the tyre model -----------------------------------------------------

    def wheel_loads(self):
        """
        Normal force under each wheel, and the ground height it is resting on.

        Load is what sets the friction limit, so a robot on two wheels of a
        four-wheel set has half the grip -- which is the behaviour that makes
        tipping and cresting a ridge feel right instead of uniform.
        """
        loads = [0.0] * len(self.contact_points)
        ground = None
        force = np.zeros(6, dtype=np.float64)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            for geom, other in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
                if geom not in self.wheel_geoms:
                    continue
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                idx = int(name[5:])
                mujoco.mj_contactForce(self.model, self.data, i, force)
                loads[idx] += abs(float(force[0]))
                z = float(con.pos[2])
                ground = z if ground is None else max(ground, z)
        return loads, ground

    def apply_tyres(self, v_cmd, omega_cmd, dt):
        """
        Turn a commanded twist into forces at the wheels, capped by grip.

        Each wheel is asked for the surface speed the twist implies at its own
        position, and the force that would cancel the difference within `dt` is
        computed and then clipped to the friction circle mu*N. That clip is the
        whole point: below it the robot tracks the command exactly and behaves
        as the kinematic model did, and above it the wheel breaks traction and
        the command is simply not met.

        Returns the fraction of available grip in use, so callers can see how
        close to the limit they are running.
        """
        loads, _ground = self.wheel_loads()
        x, y, z, yaw = self.pose
        fwd = forward_axis(yaw)
        rgt = right_axis(yaw)

        lin = np.array(self.data.qvel[0:3], dtype=np.float64)
        ## Free-joint angular velocity is in the body frame; only the yaw
        ## component matters for a wheel on the ground.
        omega_world = np.array([0.0, 0.0, float(self.data.qvel[5])])
        com = np.array(self.data.xipos[self.body_id], dtype=np.float64)

        ## Cleared each tick: mj_applyFT accumulates into the generalised force
        ## vector, so a stale entry would keep pushing after the command stopped.
        self.data.qfrc_applied[:] = 0.0
        demanded = 0.0
        available = 0.0

        for i, (ox, oy) in enumerate(self.contact_points):
            load = loads[i]
            ## Airborne wheels make no force at all -- a spinning wheel in the
            ## air does not push the robot, and pretending otherwise is how a
            ## simulated robot drives up its own front bumper.
            if load <= 1e-9:
                continue

            ## Where this wheel touches, in world space. Taken from the solver's
            ## own geom position rather than rebuilt from the body pose, so it
            ## stays correct when the robot is pitched or rolled. The contact is
            ## a radius below the sphere's centre, and getting that last term
            ## wrong applies drive force *above* the ground, which pitches the
            ## robot onto its nose and unloads the wheels that were driving it.
            centre = np.array(self.data.geom_xpos[self.wheel_ids[i]],
                              dtype=np.float64)
            point = np.array([centre[0], centre[1], centre[2] - self.wheel_radius])

            ## Velocity the robot actually has at that point.
            v_point = lin + np.cross(omega_world, point - com)
            v_long = float(np.dot(v_point, fwd))
            v_lat = float(np.dot(v_point, rgt))

            ## Velocity the commanded twist implies at this wheel. For a twist
            ## (v, omega) the body-frame velocity at (ox, oy) is
            ## (-omega*oy, v + omega*ox): a wheel offset right by ox rolls faster
            ## in a positive (CCW) turn, and a wheel ahead of centre by oy is
            ## legitimately swinging sideways.
            ##
            ## That lateral term is not optional. Driving lateral velocity to
            ## zero -- the obvious reading of "tyres resist sideways slide" --
            ## makes every wheel fight the turn the robot was asked to make, and
            ## the result is a robot that tracks speed perfectly and barely
            ## steers. What the tyre resists is lateral velocity the twist does
            ## not account for.
            target_long = v_cmd + omega_cmd * ox
            target_lat = -omega_cmd * oy

            ## The force that pulls slip toward zero over `tyre_tau`. Mass is
            ## shared between the wheels actually carrying load, so a robot up on
            ## two wheels does not get four wheels' worth of shove.
            share = self.mass / max(1, sum(1 for l in loads if l > 1e-9))
            f_long = share * (target_long - v_long) / self.tyre_tau
            f_lat = share * (target_lat - v_lat) / self.tyre_tau

            limit = self.mu * load
            magnitude = math.hypot(f_long, f_lat)
            demanded += magnitude
            available += limit
            if magnitude > limit and magnitude > 1e-12:
                ## Saturated: keep the direction, lose the surplus. This is the
                ## wheel spinning, and it is why the robot does not accelerate.
                scale = limit / magnitude
                f_long *= scale
                f_lat *= scale

            force = np.array([f_long * fwd[0] + f_lat * rgt[0],
                              f_long * fwd[1] + f_lat * rgt[1],
                              0.0])
            ## Applied at the contact point rather than the centre of mass, so
            ## the force also produces the yaw moment that turns the robot --
            ## applying it at the CoM would give a robot that drives but cannot
            ## steer, because a differential drive turns by *unequal* forces.
            mujoco.mj_applyFT(self.model, self.data, force, np.zeros(3),
                              point, self.body_id, self.data.qfrc_applied)

        if available <= 1e-12:
            return 0.0
        return min(1.0, demanded / available)

    # -- reading the world back --------------------------------------------

    def read_contacts(self):
        """
        Summarise this tick's contacts: what we are standing on, what we hit.

        A wheel touching the ground is support; anything touching the hull is an
        obstacle. Separating them by geom rather than by surface angle is what
        lets a robot drive up a steep ramp on its wheels while still being
        stopped by a wall its bumper meets.
        """
        loads, ground = self.wheel_loads()
        info = Contact()
        info.wheel_load = sum(loads)
        info.grounded = info.wheel_load > 1e-9
        info.ground_z = ground
        info.airborne = not info.grounded

        force = np.zeros(6, dtype=np.float64)
        worst = 0.0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            hull_side = None
            if con.geom1 in self.hull_geoms:
                hull_side, other = 1, con.geom2
            elif con.geom2 in self.hull_geoms:
                hull_side, other = 2, con.geom1
            if hull_side is None:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, force)
            magnitude = abs(float(force[0]))
            if magnitude <= worst:
                continue
            worst = magnitude
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other)
            info.geom = name
            info.object = self.geom_objects.get(name)
            info.force = magnitude
            info.distance = float(con.dist)
            ## Contact frame's first row is the normal, pointing from geom1 to
            ## geom2. Flip it when the hull is geom1 so the normal always points
            ## back at the robot, which is the convention RayContact reports.
            normal = np.array(con.frame[0:3], dtype=np.float64)
            if hull_side == 1:
                normal = -normal
            info.normal = tuple(float(n) for n in normal)

        info.blocked = worst >= self.blocked_threshold
        return info, loads

    # -- the interface ------------------------------------------------------

    def resolve(self, drive, start, target, velocity, dt):
        """
        Integrate one step of real dynamics and report where the robot got to.

        `target` is ignored as a *position* -- that is the difference between
        this backend and the kinematic one. The drive model's commanded twist is
        applied through the tyres, the solver decides what happens, and the pose
        that comes back is the consequence rather than a correction of what was
        asked for.
        """
        self.ensure_built()
        if dt <= 0:
            return start, velocity, None

        v_cmd, omega_cmd = velocity

        ## If something outside the contact model moved the robot -- a teleport,
        ## a reset, a scripted pose -- MuJoCo has to be told. Detecting it by
        ## comparing against what we last returned means normal driving leaves
        ## the solver's own state alone, which is what preserves momentum.
        if self.last_pose is None or not self.pose_matches(start, self.last_pose):
            self.set_pose(*start, keep_velocity=self.last_pose is not None)

        ## Substeps must cover *exactly* dt. Rounding to a whole number of
        ## fixed-size steps quietly integrates less time than the caller asked
        ## for -- at dt=1/60 and timestep=0.002 it is 8 steps of 0.002 for a
        ## 0.016667 request, a 4% time deficit that shows up as a robot whose
        ## reported velocity is right but whose odometry drifts short forever.
        ## So the count is rounded up and the step size shrunk to fit, which
        ## keeps the integration at or finer than the requested resolution.
        substeps = min(self.max_substeps,
                       max(1, int(math.ceil(dt / self.timestep - 1e-9))))
        self.model.opt.timestep = dt / substeps
        traction = 0.0
        for _ in range(substeps):
            traction = max(traction, self.apply_tyres(v_cmd, omega_cmd, dt))
            mujoco.mj_step(self.model, self.data)

        ## Applied forces are per-tick, not persistent: leaving them set would
        ## keep pushing the robot after the command stopped.
        self.data.qfrc_applied[:] = 0.0

        info, _loads = self.read_contacts()
        info.traction = traction

        x, y, z, yaw = self.pose
        lin = self.data.qvel[0:3]
        fwd = forward_axis(yaw)
        v = float(lin[0] * fwd[0] + lin[1] * fwd[1])
        omega = float(self.data.qvel[5])
        if abs(v) < self.stiction:
            v = 0.0

        ## Sliding rather than driving: grip is saturated but the robot is still
        ## moving. Reported for parity with RayContact, earned rather than
        ## inferred from a ray normal.
        info.slid = info.traction >= 0.999 and abs(v) > self.stiction

        pose = (x, y, z, yaw)
        self.last_pose = pose
        return pose, (v, omega), info

    @staticmethod
    def pose_matches(a, b, tol=1e-6):
        return all(abs(p - q) < tol for p, q in zip(a[:3], b[:3])) and \
            abs(math.atan2(math.sin(a[3] - b[3]), math.cos(a[3] - b[3]))) < tol

    # -- Blender ------------------------------------------------------------

    @classmethod
    def from_blender(cls, robot=None, objects=None, ignore=(), ground_z=None,
                     scene=None, **kw):
        """
        Build a physics world from what is actually in the Blender scene.

        Static geometry is taken as boxes from each object's world-space bounding
        box, which is exact for the walls, ramps and blocks the corpus generator
        makes and an over-approximation for anything organic. `add_mesh` is there
        for the cases where that is not good enough.

        Imported lazily so this module stays usable headlessly: nothing above
        this point needs Blender, and this is the only method that does.
        """
        import bpy
        scene = scene or bpy.context.scene
        skip = {getattr(o, 'name', o) for o in ignore}
        if robot is not None:
            skip |= {p.name for p in robot.parts()}
            kw.setdefault('contact_points',
                          [(w.x, w.y) for w in robot.wheel_list] or None)
            kw.setdefault('wheel_radius', robot.wheel_radius)
            kw.setdefault('size', tuple(robot.size))

        self = cls(scene=scene, **kw)
        if ground_z is not None:
            self.add_ground(z=ground_z)

        for obj in (objects if objects is not None else scene.objects):
            if obj.name in skip or obj.type != 'MESH':
                continue
            ## World-space centre and extent: `dimensions` already has the
            ## object's scale baked in, which local bound_box does not.
            centre = obj.matrix_world.translation
            self.add_box(obj.name, (centre.x, centre.y, centre.z),
                         tuple(obj.dimensions), yaw=obj.rotation_euler.z,
                         obj=obj)
        return self.build()

    def sync_to_blender(self, root):
        """Write the solver's pose back onto a Blender object, tilt included."""
        x, y, z, yaw = self.pose
        root.location = (x, y, z)
        q = self.data.qpos[3:7]
        root.rotation_euler = self.quat_to_euler(q)
        return root

    @staticmethod
    def quat_to_euler(q):
        """(w, x, y, z) to Blender's XYZ euler, matching Rz @ Ry @ Rx."""
        w, x, y, z = (float(v) for v in q)
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr, cosr)
        sinp = 2.0 * (w * y - z * x)
        pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        return (roll, pitch, math.atan2(siny, cosy))
