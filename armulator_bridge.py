"""
Gating robotsim's plant against armulator's register-level hardware models.

robotsim's drive seam is `set_wheel_speeds(left, right)` in metres per second, and its
lidar seam is a Blender ray cast returning metres. Both are *value* seams: they say what
the hardware ends up doing without saying how firmware asked for it. That is the right
level for working on the robot and the wrong level for asking whether a driver is
correct, exactly as `firmware.py` argues about hostsim.

This module closes the same gap one layer lower. armulator models the PCA9685's
registers, the TB6612FNG truth table, the motor that integrates the result, and the
RPLIDAR wire protocol. Running robotsim's own integrator with that stack spliced into
the middle isolates what the register layer adds:

    report = compare_drive([(1.0, 0.5, 0.0)])
    report.max_speed_error        # m/s the register path lags the value path
    report.final_position_error   # metres apart after the profile

WHY THIS IS OFFLINE AND NOT IN THE TICK LOOP
--------------------------------------------
`firmware.py` is right that armulator belongs offline, but for a narrower reason than it
states. The ~80,000x figure is about **AArch64 instruction emulation** -- `ArmV8`
stepping opcodes with an MMU. The peripheral models used here execute no instructions:
they are float arithmetic, and cost single-digit microseconds per tick against a 60Hz
budget of 16,667us.

So the cost is not why this is a gate. The *reason* is that a gate answers a different
question than a simulation does. In the loop, armulator's motor model would simply
become the plant, and agreement with robotsim would be untestable because there would be
nothing left to disagree with. Kept separate, the two are independent implementations of
the same physical claim, and their divergence is a measurement. Splicing them together
would destroy the only thing being measured.

WHAT DIVERGENCE MEANS
---------------------
Perfect agreement is the wrong expectation and would indicate the gate is not wired up.
The register path *should* differ, in ways that are all real hardware behaviour:

  * **Spin-up lag.** A commanded wheel speed is reached instantly at the value seam and
    approaches exponentially through a real motor.
  * **Duty quantisation.** The PCA9685 has 12-bit outputs, so a commanded speed is
    rounded to one part in 4095.
  * **Stall deadband.** Below `stall_drive` the shaft does not turn at all, where the
    value seam happily moves the robot at a crawl.
  * **Channel routing.** M2 and M4 have their direction channels in the opposite order
    to M1 and M3. A driver that assumes a uniform pattern drives backwards, and this is
    where that shows up as a sign error rather than as a code review comment.

A gate asserts those stay within a budget, not that they are zero.

BLENDER
-------
Nothing here imports `bpy` at module scope. The lidar adapter takes a ray-cast callable,
so it works under Blender with the real scene and works headless with any function of
the same shape -- which is what makes the lidar half of this testable at all on a machine
without Blender.
"""

import math
import os
import sys

try:
    import firmware
except ImportError:                      # imported as part of a package
    from . import firmware


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def available():
    """
    True when armulator is cloned beside robotsim and importable.

    Reported rather than raised, matching `firmware.available()`, so a test can skip on
    a machine without the optional checkout instead of failing on one.
    """
    root = firmware.armulator_root()
    if not root:
        return False
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import armulator.peripherals.motor_hat            # noqa: F401
        import armulator.sensors                          # noqa: F401
    except ImportError:
        return False
    return True


def why_unavailable():
    """A sentence explaining what is missing, for a skip message."""
    root = firmware.armulator_root()
    if not root:
        return ('armulator is not cloned beside robotsim (expected %s) -- '
                'git clone https://github.com/crustos/armulator.git'
                % os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), 'armulator'))
    return ('armulator is at %s but did not import; it may predate '
            'armulator.sensors' % root)


def _armulator():
    """Import the pieces used here, raising a useful error if they are missing."""
    if not available():
        raise RuntimeError(why_unavailable())
    from armulator.peripherals.motor_hat import MotorHat
    from armulator.peripherals.pca9685 import LED0_ON_L, MODE1, MODE1_AI
    from armulator.sensors import Lidar
    return MotorHat, Lidar, (LED0_ON_L, MODE1, MODE1_AI)


# ---------------------------------------------------------------------------
# a pose object that is not a Blender object
# ---------------------------------------------------------------------------

class _Axis:
    __slots__ = ('x', 'y', 'z')

    def __init__(self):
        self.x = self.y = self.z = 0.0


class HeadlessRoot:
    """
    The duck-typed subset of a Blender object that `DriveBase` actually touches.

    `DriveBase` reads and writes `root.location.x/y/z` and `root.rotation_euler.z` and
    nothing else, so a gate can drive robotsim's real integrator without Blender running.
    That matters: the alternative is reimplementing `DriveBase.step`, and a gate built on
    a reimplementation of the thing it is gating proves nothing.
    """

    def __init__(self):
        self.location = _Axis()
        self.rotation_euler = _Axis()

    @property
    def pose(self):
        return (self.location.x, self.location.y, self.rotation_euler.z)

    def __repr__(self):
        x, y, yaw = self.pose
        return '<HeadlessRoot (%.3f, %.3f) yaw=%.3f>' % (x, y, yaw)


# ---------------------------------------------------------------------------
# the motor seam: PWM registers in, wheel speeds out
# ---------------------------------------------------------------------------

class MotorPlant:
    """
    Two wheels driven through armulator's full I2C-to-shaft path.

    :param wheel_radius: metres
    :param gear_ratio: motor turns per wheel turn
    :param free_speed: motor shaft speed at full drive, rev/s
    :param positions: which HAT motor positions the wheels are on

    The default positions are 1 and 2 deliberately. They have their direction channels
    in opposite orders on the real HAT, so a bridge that got the routing wrong drives one
    wheel backwards and the gate reports a robot spinning on the spot rather than
    driving -- which is a failure you can read, unlike a register diff.
    """

    def __init__(self, wheel_radius=0.1, gear_ratio=1.0, free_speed=5.0,
                 positions=(1, 2), spin_up=0.15, stall_drive=0.08):
        MotorHat, _, registers = _armulator()
        self.LED0_ON_L, MODE1, MODE1_AI = registers

        self.wheel_radius = wheel_radius
        self.gear_ratio = gear_ratio
        self.free_speed = free_speed
        self.left_position, self.right_position = positions

        self.hat = MotorHat()
        # Wake the controller and turn auto-increment on, as a driver's init does.
        self.hat.controller.write([MODE1, MODE1_AI])
        self.left = self.hat.attach_dc_motor(
            self.left_position, free_speed=free_speed, spin_up=spin_up,
            stall_drive=stall_drive)
        self.right = self.hat.attach_dc_motor(
            self.right_position, free_speed=free_speed, spin_up=spin_up,
            stall_drive=stall_drive)

    # -- unit conversion ----------------------------------------------------

    @property
    def max_wheel_speed(self):
        """Wheel surface speed at full drive, m/s. Commands above this clip."""
        return self.shaft_to_surface(self.free_speed)

    def shaft_to_surface(self, rev_per_second):
        """Motor shaft rev/s to wheel contact-patch m/s, through the gearbox."""
        return rev_per_second / self.gear_ratio * 2.0 * math.pi * self.wheel_radius

    def surface_to_shaft(self, metres_per_second):
        """The inverse: what the shaft must do for the wheel to travel this fast."""
        return metres_per_second * self.gear_ratio / (2.0 * math.pi * self.wheel_radius)

    # -- the driver ---------------------------------------------------------

    def _set_channel(self, channel, duty):
        """Program one PWM channel: four registers from a zero start, as a driver does."""
        count = int(duty * 4095)
        self.hat.controller.write(
            [self.LED0_ON_L + 4 * channel, 0, 0, count & 0xFF, count >> 8])

    def command(self, position, surface_speed):
        """
        Drive one wheel at a signed speed by writing PCA9685 registers.

        This is the artifact under test: an open-loop driver that converts a speed into
        a duty cycle and a direction, exactly as embedded code would. It reads the
        channel map from the HAT rather than assuming one, which is the bug the M2/M4
        routing is there to catch.
        """
        pwm, in2, in1 = self.hat.channels[position].channels
        shaft = self.surface_to_shaft(surface_speed)
        duty = max(-1.0, min(1.0, shaft / self.free_speed))

        self._set_channel(in1, 1.0 if duty > 0 else 0.0)
        self._set_channel(in2, 1.0 if duty < 0 else 0.0)
        self._set_channel(pwm, abs(duty))

    def set_wheel_speeds(self, left, right):
        """Command both wheels, mirroring `DifferentialDrive.set_wheel_speeds`."""
        self.command(self.left_position, left)
        self.command(self.right_position, right)
        return self

    def brake(self):
        """Both direction pins high: shorted and stopping fast, not coasting."""
        for position in (self.left_position, self.right_position):
            pwm, in2, in1 = self.hat.channels[position].channels
            self._set_channel(in1, 1.0)
            self._set_channel(in2, 1.0)
        return self

    # -- running ------------------------------------------------------------

    def advance(self, dt):
        """Let the motors turn for `dt` seconds."""
        self.hat.advance(dt)

    def wheel_speeds(self):
        """What the shafts are actually doing, as wheel m/s."""
        return (self.shaft_to_surface(self.left.speed),
                self.shaft_to_surface(self.right.speed))

    def wheel_distances(self):
        """How far each wheel has travelled, in metres. The encoder robotsim lacks."""
        return (self.shaft_to_surface(self.left.position),
                self.shaft_to_surface(self.right.position))

    def __repr__(self):
        left, right = self.wheel_speeds()
        return '<MotorPlant L=%.3f R=%.3f m/s>' % (left, right)


# ---------------------------------------------------------------------------
# the lidar seam: a robotsim world behind an armulator sensor
# ---------------------------------------------------------------------------

class SceneWorld:
    """
    Stands in for `armulator.sensors.Room`, backed by robotsim's ray caster.

    armulator's Lidar asks its world exactly one question --
    `range_at((x, y), bearing_radians) -> metres or None` -- so anything answering that
    is a drop-in world. Supplying this one instead of a `Room` means firmware gets
    RPLIDAR packets describing the actual Blender scene.

    :param cast: callable `(origin_xy, bearing_radians) -> distance or None`
    :param z: height of the scan plane, passed through to the cast

    The callable is injected rather than `bpy` being imported here, for two reasons.
    It keeps this module importable without Blender, and it makes the lidar half of the
    gate runnable against known geometry -- which is the only way to tell a protocol bug
    from a ray-casting bug.
    """

    def __init__(self, cast, z=0.0, name='scene'):
        if not callable(cast):
            raise TypeError('cast must be callable, got %r' % type(cast).__name__)
        self.cast = cast
        self.z = z
        self.name = name

    def range_at(self, origin, bearing):
        return self.cast(origin, bearing)

    @classmethod
    def from_blender(cls, scene=None, depsgraph=None, z=0.0, ignore=()):
        """
        Build a world from the live Blender scene.

        Imports `bpy` lazily, so importing this module on a machine without Blender is
        still fine. Delegates to `sensors.cast_ignoring`, so the ranges the firmware
        eventually sees come from the same ray caster robotsim's own Lidar uses -- which
        is what makes any disagreement meaningful.
        """
        import bpy                                        # noqa: F401  (lazy)
        import sensors

        scene = scene or bpy.context.scene
        depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()

        def cast(origin, bearing):
            direction = sensors.ray_direction(bearing)
            start = (origin[0], origin[1], z)
            hit = sensors.cast_ignoring(scene, depsgraph, start, direction,
                                        float('inf'), ignore=ignore)
            distance = hit[1] if isinstance(hit, tuple) else hit
            if distance is None or distance == float('inf'):
                return None
            return distance

        return cls(cast, z=z, name='blender')

    def __repr__(self):
        return '<SceneWorld %s z=%.2f>' % (self.name, self.z)


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------

class DriveReport:
    """
    What the register path and the value path each did, and how far apart they are.

    Divergence is reported rather than judged. What counts as acceptable depends on the
    robot, so `passes()` takes the budget from the caller instead of baking one in.
    """

    def __init__(self, samples, reference_root, plant_root):
        #: (t, v_reference, v_register, left_error, right_error) per step.
        self.samples = samples
        self.reference_pose = reference_root.pose
        self.plant_pose = plant_root.pose

    @property
    def max_speed_error(self):
        """
        Largest single-wheel disagreement in m/s, over the whole profile.

        Diagnostic rather than pass/fail: at a step change this is dominated by the
        spin-up transient and lands near the commanded speed no matter how correct the
        implementation is. See :meth:`passes`.
        """
        if not self.samples:
            return 0.0
        return max(max(abs(s[3]), abs(s[4])) for s in self.samples)

    @property
    def final_speed_error(self):
        """Steady-state disagreement once the motors have settled. The gating metric."""
        if not self.samples:
            return 0.0
        last = self.samples[-1]
        return max(abs(last[3]), abs(last[4]))

    @property
    def final_position_error(self):
        """Distance in metres between where the two paths think the robot ended."""
        ax, ay, _ = self.reference_pose
        bx, by, _ = self.plant_pose
        return math.hypot(ax - bx, ay - by)

    @property
    def final_heading_error(self):
        """Absolute yaw disagreement in radians, wrapped to [0, pi]."""
        _, _, a = self.reference_pose
        _, _, b = self.plant_pose
        return abs((a - b + math.pi) % (2 * math.pi) - math.pi)

    def passes(self, speed=0.05, position=0.5, heading=math.radians(15)):
        """
        Whether every divergence sits inside the supplied budget.

        Note this uses :attr:`final_speed_error`, not :attr:`max_speed_error`. At a step
        change the register path starts from rest while the value path is instantly at
        speed, so the peak disagreement is approximately the commanded speed itself --
        for any correct implementation. Gating on it would fail everything and could
        only be made to pass by removing the spin-up model, which is the one piece of
        physics worth having. What a gate can meaningfully ask is whether the two paths
        *agree once settled*, which is what the steady-state error measures.
        """
        return (self.final_speed_error <= speed
                and self.final_position_error <= position
                and self.final_heading_error <= heading)

    def format(self):
        return '\n'.join([
            'drive conformance over %d steps' % len(self.samples),
            '  max wheel speed error   %.4f m/s' % self.max_speed_error,
            '  final wheel speed error %.4f m/s' % self.final_speed_error,
            '  final position error    %.4f m' % self.final_position_error,
            '  final heading error     %.4f rad' % self.final_heading_error,
            '  reference pose  (%.3f, %.3f) yaw %.3f' % self.reference_pose,
            '  register pose   (%.3f, %.3f) yaw %.3f' % self.plant_pose,
        ])

    def __repr__(self):
        return ('<DriveReport %d steps, %.3f m apart>'
                % (len(self.samples), self.final_position_error))


class LidarReport:
    """
    How faithfully the register and protocol path carries the world's geometry.

    The comparison is against the same world the sensor was given, so a disagreement is
    the encoding, not the geometry. Two separate quantisations contribute, and the
    second is easy to miss:

      * **Distance** travels as q2 millimetres, so ranges round to 0.25mm.
      * **Angle** travels as q6 degrees, so a decoded bearing is up to 1/128 of a degree
        away from the one the beam was actually taken at.

    The angular term is usually the larger of the two, because its effect on range is
    ``|dr/dtheta| * dtheta`` and that gradient is unbounded. A beam striking a flat
    surface at grazing incidence moves its contact point a long way for a very small
    change in bearing; in a corridor it dwarfs the distance rounding by an order of
    magnitude. Budgeting only for the distance quantisation makes a correct
    implementation look broken.

    So each range is checked against the *bracket* of values the world takes across the
    angular uncertainty, rather than against a single expected number. A reading inside
    that bracket is consistent with the geometry and scores zero error; the metric is
    how far outside it any reading falls.
    """

    #: Metres of error the q2 millimetre distance encoding can introduce.
    QUANTISATION = 0.000125

    #: Angle least-significant bit, in degrees. Angles travel as q6, so 1/64 degree.
    ANGLE_LSB = 1.0 / 64.0

    #: Worst-case bearing error from rounding to that grid: half an LSB either way.
    ANGLE_QUANTISATION = ANGLE_LSB / 2.0

    def __init__(self, pairs, dropped, decoded, nodes):
        #: (bearing_degrees, expected, decoded, bracket_low, bracket_high) per return.
        self.pairs = pairs
        #: Measurements with no return, which carry no range to compare.
        self.dropped = dropped
        self.decoded = decoded
        self.nodes = nodes

    @staticmethod
    def _excess(value, low, high):
        """How far outside [low, high] a value falls; zero when inside."""
        if value < low:
            return low - value
        if value > high:
            return value - high
        return 0.0

    @property
    def max_range_error(self):
        """
        Worst departure from what the geometry allows, in metres.

        Zero means every reading was consistent with the world once both quantisations
        are accounted for.
        """
        if not self.pairs:
            return 0.0
        return max(self._excess(got, low, high)
                   for _, _, got, low, high in self.pairs)

    @property
    def max_raw_error(self):
        """
        Worst naive difference, ignoring angular uncertainty.

        Reported because it is the number someone will compute by hand and be alarmed
        by; seeing it alongside the bracketed figure is what explains the gap.
        """
        if not self.pairs:
            return 0.0
        return max(abs(expected - got) for _, expected, got, _, _ in self.pairs)

    @property
    def compared(self):
        return len(self.pairs)

    def passes(self, tolerance=None):
        """
        Whether every decoded range is consistent with the world it came from.

        Defaults to the distance quantisation alone, because the angular term is already
        absorbed into each reading's bracket. This is therefore a strict check that the
        register and wire path introduced nothing of its own.
        """
        if tolerance is None:
            tolerance = self.QUANTISATION * 2
        return self.max_range_error <= tolerance and self.compared > 0

    def format(self):
        return '\n'.join([
            'lidar conformance over %d nodes' % self.nodes,
            '  ranges compared      %d' % self.compared,
            '  no-return samples    %d' % self.dropped,
            '  max error vs bracket %.6f m' % self.max_range_error,
            '  max naive difference %.6f m  (dominated by angle quantisation)'
            % self.max_raw_error,
            '  distance resolution  %.6f m' % self.QUANTISATION,
            '  angle resolution     %.6f deg (half of a %.6f deg LSB)'
            % (self.ANGLE_QUANTISATION, self.ANGLE_LSB),
        ])

    def __repr__(self):
        return ('<LidarReport %d ranges, max error %.6f m>'
                % (self.compared, self.max_range_error))


# ---------------------------------------------------------------------------
# the gates
# ---------------------------------------------------------------------------

def compare_drive(profile, dt=1.0 / 60.0, track=0.5, wheel_radius=0.1,
                  plant=None, **drive_kwargs):
    """
    Run one commanded profile through both paths and report the difference.

    :param profile: list of `(seconds, v, omega)` segments, m/s and rad/s
    :param plant: a configured :class:`MotorPlant`, or None to build a default
    :returns: :class:`DriveReport`

    Both paths use robotsim's own `DifferentialDrive.step`, with identical parameters
    and identical commands. The *only* difference is that the reference path feeds
    commanded wheel speeds straight into the integrator, while the register path routes
    them through PCA9685 writes and reads back what the shafts actually did. Anything
    that diverges is therefore attributable to the register layer and to nothing else,
    which is the entire point of holding the integrator constant.
    """
    try:
        import drive as drive_module
    except ImportError:                                   # pragma: no cover
        from . import drive as drive_module

    plant = plant or MotorPlant(wheel_radius=wheel_radius)

    reference_root, plant_root = HeadlessRoot(), HeadlessRoot()
    reference = drive_module.DifferentialDrive(
        reference_root, wheels=None, wheel_radius=wheel_radius, track=track,
        **drive_kwargs)
    measured = drive_module.DifferentialDrive(
        plant_root, wheels=None, wheel_radius=wheel_radius, track=track,
        **drive_kwargs)

    samples = []
    clock = 0.0
    for seconds, v, omega in profile:
        steps = max(1, int(round(seconds / dt)))
        for _ in range(steps):
            # The command a controller issues, identical for both paths.
            half = 0.5 * omega * track
            left_cmd, right_cmd = v - half, v + half

            reference.set_wheel_speeds(left_cmd, right_cmd)
            reference.step(dt)

            plant.set_wheel_speeds(left_cmd, right_cmd)
            plant.advance(dt)
            left_actual, right_actual = plant.wheel_speeds()
            measured.set_wheel_speeds(left_actual, right_actual)
            measured.step(dt)

            clock += dt
            samples.append((clock, v, 0.5 * (left_actual + right_actual),
                            left_actual - left_cmd, right_actual - right_cmd))

    return DriveReport(samples, reference_root, plant_root)


def compare_lidar(world, seconds=1.0, dt=1.0 / 60.0, pose=(0.0, 0.0, 0.0),
                  sample_rate=2000, max_range=6.0, min_range=0.15, lidar=None):
    """
    Check that what comes off the RPLIDAR wire is the world the sensor was given.

    :param world: anything with `range_at((x, y), bearing)` -- a `Room`, a
        :class:`SceneWorld`, or a stub
    :returns: :class:`LidarReport`

    Ranges are decoded back out of the UART's receive FIFO rather than read from the
    model, so the comparison covers duty routing, the spin, the q6 angle encoding, the
    q2 distance encoding and the framing bits. Reading the model's own measurement list
    would skip all of that and compare the geometry against itself.
    """
    if not available():
        raise RuntimeError(why_unavailable())
    from armulator.peripherals.motor_hat import MotorHat
    from armulator.peripherals.pca9685 import LED0_ON_L, MODE1, MODE1_AI
    from armulator.peripherals.uart_pl011 import Pl011Uart
    from armulator.sensors import CMD_SCAN, SYNC, Lidar, decode_measurement

    hat = MotorHat()
    hat.controller.write([MODE1, MODE1_AI])
    uart = Pl011Uart()

    lidar = lidar or Lidar(room=world, pose=pose, sample_rate=sample_rate,
                           max_range=max_range, min_range=min_range)
    lidar.room = world
    lidar.attach_to(hat, 4).connect(uart)

    # Spin the head by writing registers, as firmware would.
    pwm, in2, in1 = hat.channels[4].channels
    for channel, duty in ((in2, 0.0), (in1, 1.0), (pwm, 1.0)):
        count = int(duty * 4095)
        hat.controller.write([LED0_ON_L + 4 * channel, 0, 0, count & 0xFF, count >> 8])

    # Ask for a scan, then drop the response descriptor so the FIFO is pure nodes.
    for byte in (SYNC, CMD_SCAN):
        for callback in uart.tx_callbacks:
            callback(byte)
    uart._rx_fifo.clear()

    steps = max(1, int(round(seconds / dt)))
    for _ in range(steps):
        hat.advance(dt)

    stream = bytes(uart._rx_fifo)
    nodes = [stream[i:i + 5] for i in range(0, len(stream) - 4, 5)]

    pairs, dropped = [], 0
    origin = (lidar.pose[0], lidar.pose[1])
    _, _, heading = lidar.pose
    # Half an LSB either side: the full span the true bearing could have occupied.
    half_step = LidarReport.ANGLE_QUANTISATION
    for node in nodes:
        measurement = decode_measurement(node)
        if measurement.distance is None:
            dropped += 1
            continue
        bearing = measurement.angle + math.degrees(heading)
        expected = world.range_at(origin, math.radians(bearing))
        if expected is None:
            dropped += 1
            continue
        # The true beam lay somewhere within half a quantisation step of the decoded
        # bearing, so the range it should have measured is anywhere in this bracket.
        # Sampling the endpoints is exact for a monotonic surface and close enough on
        # a curved one, where the gradient is small anyway.
        candidates = [expected]
        for offset in (-half_step, half_step):
            edge = world.range_at(origin, math.radians(bearing + offset))
            if edge is not None:
                candidates.append(edge)
        low = min(candidates) - LidarReport.QUANTISATION
        high = max(candidates) + LidarReport.QUANTISATION
        pairs.append((measurement.angle, expected, measurement.distance, low, high))

    return LidarReport(pairs, dropped, len(nodes), len(nodes))
