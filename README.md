# robotsim — a Blender-based robot simulation platform

<img src="/examples/3_arm_demo/Animation/gifmaker_me.gif"/>

## What this is

A simulation platform for **mobile robots with manipulators**, built on Blender.

You describe a robot in Python — a wheeled base, one or more arms loaded from
`.blend` files, a mast of cameras — and the platform gives you a fixed-timestep
loop in which you can drive the base, command the joints, render what each camera
sees, and bake the whole run onto the Blender timeline as an ordinary animation.

The target application is **vision-driven control**. The camera renders from each
robot are intended as the input tensor for a neural network trained in PyTorch;
the trained network is then deployed to an onboard microcontroller (a Jetson Nano
class device) where it drives the wheel motors directly. Blender is the world
model, the renderer, and the ground-truth source for that training loop.

## Why Blender

Blender already ships the hard parts: an IK solver, an armature system with joint
limits, a renderer, a timeline with interpolation, and a Python API over all of
it. Rigging a new arm is a modelling task rather than a URDF-authoring task, and
the result is immediately visualisable. The trade is that Blender is an animation
tool, not a physics engine — so this platform is deliberately *kinematic* today,
with the seams left in the right places to bolt on an external solver later.

## Quick start

```bash
git clone https://github.com/crustos/blender_manipulator_motion_demo
cd blender_manipulator_motion_demo
make install        # chmod the entry points, apt-get install blender
make test_all       # run the full test suite headless
```

Scripts are sh/Python polyglots: they exec Blender on themselves, so they run
directly.

```bash
./robotsim.py                 # interactive, with a UI
./headless.py my_scene.py     # background, runs your script inside Blender
```

Any `.py` passed after `--` is exec'd with the platform's globals available, so a
script can use `Robot`, `RobotSim`, `Arm` and friends without importing anything.

## A minimal simulation

```python
#!../headless.py

bot = Robot(size=(1, 1, 0.1), wheels=4, drive='differential')
rec = RobotSim.record()          # bake this run onto the timeline

bot.drive.drive(1.0, 0.4)        # 1 m/s forward, 0.4 rad/s yaw
arm = bot.arms[0]

@RobotSim
def tick(dt):
    arm.set_tip(location=(0.3, 0.4, 0.5))     # IK: move the tool tip
    views = bot.sample_cameras(frame=RobotSim.frame)   # renders only when due
    if RobotSim.ticks >= 100:
        RobotSim.stop()

while RobotSim.callbacks:
    RobotSim.update()
```

## Architecture

| Module | Depends on bpy | Purpose |
| --- | --- | --- |
| `robotsim.py` | yes | Entry point and assembly layer. Scene primitives, `.blend` loading, the `Robot` model, `RobotSimpleSim` fixed-timestep loop, camera rendering. |
| `kinematics.py` | yes | `Arm` and `Joint`. Joint-space read/write over an armature, IK muting, joint limits, tool-tip control. |
| `drive.py` | **no** | Base motion and wheel layout. `Wheel`, `wheel_layout()`, `DifferentialDrive` and `AckermannDrive`, integrated with a real timestep, plus the `ContactModel` seam. Pure maths, unit-testable outside Blender. |
| `recorder.py` | yes | Binds sim ticks to scene frames and bakes state into keyframes. |
| `robotsim.SensorRig` | yes | Multi-pass capture. Owns the compositor graph that turns one render into RGB, depth and segmentation files. |
| `sensors.py` | yes | `Lidar` and `LidarScan`. Ray-cast ranging with per-beam labels, independent of the render path. |
| `contact.py` | yes | `RayContact`. Ground following, collision and slip by ray casting, behind the `drive.ContactModel` seam. |
| `npr.py` | yes | Non-photorealistic line art. Flattens the scene and renders structure only. |
| `randomize.py` | yes | Seeded domain randomisation: layout, materials, lighting, viewpoint. |
| `dataset.py` | **no** | Corpus writer. Aligned samples, a manifest that explains them, and verification. |
| `perception.py` | **no** | Stage 1: learns photograph to line drawing. Pure numpy, no Blender, verified against a numerical gradient check. |
| `telemetry.py` | **no** | Records channels per tick and draws them as a matplotlib panel with the firmware console attached. Optional: only drawing needs matplotlib. |
| `firmware.py` | **no** | Real C/C++ firmware in the loop via crust's hostsim, and the `Network` bus between boards. Optional: does nothing unless crust is cloned beside robotsim. |
| `ros2/joint_export.py` | yes | Exports a recorded arm motion as a ROS2-style joint trajectory (JSON or text). |
| `ros2/blender_to_text.py` | yes | The original upstream exporter, kept for the hand-animated workflow. |

## Core concepts

### Frame convention

Derived from how `Robot` is built — the `front` camera and `FRONT.HUB` both face
`+Y`, so:

| Axis | Direction |
| --- | --- |
| `+Y` | forward |
| `+X` | right |
| `+Z` | up |
| `rotation_euler.z` | yaw, measured from `+Y` |

### Reading joints vs writing joints

This asymmetry causes most of the confusion when scripting Blender armatures, so
it is worth stating plainly:

- **Reading works in any mode.** `arm.angles` recovers each joint angle from the
  evaluated bone matrices relative to the rest pose. Reading
  `pose_bone.rotation_euler` while IK drives the chain returns the *authored*
  value, not the solved one.
- **Writing requires IK off.** The solver overwrites `rotation_euler` on the next
  depsgraph evaluation. `arm.set_angles()` mutes the IK constraints for you;
  `arm.set_tip()` turns them back on.

The same rule governs recording: with IK live the recorder keyframes the tool-tip
empty and lets the solver re-derive the chain on playback; with IK off it bakes
the joint rotations directly.

### Timestep and the timeline

`RobotSim.dt` is *simulated* time per tick, not wall-clock — the loop runs as
fast as it can and motion advances by exactly `dt`, so a run is reproducible no
matter how slow rendering is. Recording maps one tick to one scene frame and sets
the scene frame rate from `dt` (stored as Blender's rational `fps/fps_base`, so
arbitrary `dt` values encode exactly). Keyframes are forced to LINEAR
interpolation: Blender's default Bezier easing would invent motion between
samples that the integrator never produced.

## Rig configuration

A `Robot` is described by its body size, its wheels and its arms. All three
accept either a shorthand or a full specification.

### Wheels

`wheels` is either a count or an explicit list of placements.

```python
Robot(wheels=4)          # two axles, left/right pairs  (the default)
Robot(wheels=2)          # one axle: a classic differential base
Robot(wheels=3)          # tricycle: a pair at the rear, one centre wheel up front
Robot(wheels=6)          # three axles
Robot(wheels=0)          # no wheels; the base still moves
```

Even counts become left/right pairs spread over `count/2` axles. Odd counts get
the same pairs plus a single unpowered centre wheel on the front axle — a caster
under differential drive, the steered wheel of a tricycle under Ackermann. Axles
are named front to rear, and the outermost two are always `FRONT` and `REAR`
because the drive models use them to find the steering wheels and to measure the
wheelbase.

Placing wheels by hand needs only a location; everything else is derived, with
the side taken from the sign of the lateral offset:

```python
Robot(wheels=[
    {'location': (-0.2, -0.4, -0.05)},                  # -> W.L.REAR
    {'location': ( 0.2, -0.4, -0.05)},                  # -> W.R.REAR
    {'location': ( 0.0,  0.9, -0.05), 'driven': False}, # -> W.C.MID, passive
])
```

Track and wheelbase are then *measured* from the wheels that exist rather than
assumed from the body box, so a narrow-track rover turns correctly.

Each wheel's surface speed is computed from its lateral offset (`v + ω·x`) rather
than by matching on its name. For a two-sided layout that recovers the commanded
left/right speeds exactly, and it is also the right answer for centre wheels and
extra axles.

### Arm mounts

`arms` accepts a bare path, a `(path, location)` pair, or a full dict:

```python
Robot(arms=[DEFAULT_ARM])                                   # centre of the front edge
Robot(arms=[DEFAULT_ARM] * 3)                               # spread across the front
Robot(arms=[(DEFAULT_ARM, (0.3, -0.2, 0.4))])               # placed
Robot(arms=[{'path': DEFAULT_ARM, 'location': (-0.3, 0.2, 0.4),
             'rotation': (0, 0, math.pi), 'parent': 'front_hub'}])
```

Several arms are laid out evenly across the front edge instead of being stacked
at one point; a single arm still lands exactly where it always did. `parent` may
be any object or the name of a robot part (`root`, `body`, `front_hub`,
`rear_hub`, `camera_hub`), so an arm can ride on a hub and inherit its motion.
Arms can also be added after construction:

```python
bot.add_arm(DEFAULT_ARM, location=(0, 0, 0.3), parent='front_hub')
```

### Cameras and capture rate

`cameras` selects the sensor set: `'all'` (the four-camera mast, the default),
`'front'` for a single forward-facing camera, `'none'`, or an explicit list.

```python
Robot(cameras='front')                       # one camera: a quarter of the work
Robot(cameras=['front', 'back'])
Robot(cameras='none')                        # blind: drives, never renders
```

`camera_interval` is how many ticks pass between captures, and defaults to 30 —
one capture per simulated second at `dt = 1/30`.

```python
@RobotSim
def tick(dt):
    bot.drive.drive(v, omega)                # every tick: cheap
    views = bot.sample_cameras()             # only when due: expensive
```

`sample_cameras()` renders only on a capture tick and returns `[]` otherwise;
`render_cameras()` ignores the interval and always renders. The first tick always
captures, since a controller needs an initial view before it has anything to
reason about. `camera_interval=0` disables capture entirely, `1` captures every
tick, and `force=True` takes a one-off grab regardless.

This is a model of the target hardware, not just a test speedup. On a Jetson-class
board the control loop runs fast on cheap sensors — wheel odometry, IMU, GPS,
lidar, proximity — and vision is the expensive sense that gets sampled far more
rarely and irregularly. A policy that only peeks at the cameras when it needs to
refresh its picture of the world is both what the power budget wants and what the
sim should be training against, so the duty cycle belongs in the platform rather
than in each script.

## Performance

Renders dominate the loop, so the defaults are chosen to make them rare and cheap
rather than to look good. Measured in background mode on one machine:

| Lever | Effect |
| --- | --- |
| Render engine | EEVEE ~2.24 s/render, Workbench ~0.08 s/render |
| Resolution | ~0 on EEVEE: 64x32 and 320x240 both cost ~2.24 s |
| Camera count | linear — one camera is a quarter of four |
| Capture interval | linear — the default 30 renders 30x less often |
| Mesh decimation | irb120 50778 → 6619 verts, ur10 87841 → 7562 |

The counter-intuitive one is resolution. In background mode EEVEE's cost is
per-render context setup, not pixels, so shrinking the image buys almost nothing.
The real levers are rendering *less often*, from *fewer cameras*, on a *cheaper
engine*. `quick_render()` still defaults to 64x32 — but to keep the input tensors
small, not to save time.

`set_render_engine('workbench')` is the single biggest win for tests that only
need to prove a camera pointed somewhere and produced pixels. It draws solid
shaded geometry with no lights, shadows or materials, so it is wrong for training
data and right for a smoke test.

Arm `.blend` files are CAD exports carrying far more detail than a 64x32 render
can show, so meshes over 3000 vertices are welded and decimated to about 1000 on
load. Pass `decimate=False` to `load_blend_objects()` to keep the source geometry
for a final high-quality render. Welding matters as much as decimating: these
exports split vertices per face — ur10's `link_4` has 17302 vertices but only
13443 triangles — and collapse decimation cannot merge across those splits, so
without a weld first it floors out around 6700 no matter what ratio it is given.

## Sensors

Depth and segmentation are render *passes*, not extra renders. The renderer
already knows the depth and the owning object of every pixel it shades, so the
sensor API is built around **one render per camera, however many modalities are
asked for** — rather than one call per sensor.

```python
bot = Robot(cameras='front', passes=('rgb', 'depth', 'segmentation'))

views = bot.capture(frame=RobotSim.frame)
views['front']['depth']          # -> /tmp/ROBOT.ROOT.front.0007.depth.exr
```

`capture()` returns `{camera: {pass: path}}` and always renders; `sample()` is
the rate-limited form that returns `{}` when the tick is not due, exactly as
`sample_cameras()` does for RGB alone. Available passes are `rgb`, `depth`,
`segmentation`, `normal` and `mist`.

| Pass | Format | Contents |
| --- | --- | --- |
| `rgb` | PNG | the colour image |
| `depth` | 32-bit EXR | metres from the camera; unhit pixels read very large |
| `segmentation` | 32-bit EXR | integer class index per pixel |
| `normal`, `mist` | 32-bit EXR | surface normals, normalised distance |

Data passes are written as float EXR rather than PNG deliberately: metric depth
and integer class indices do not survive an 8-bit colour-managed image.

### Segmentation labels

Robot parts are labelled by class on construction — `background` 0, `ground` 1,
`body` 2, `wheel` 3, `hub` 4, `arm` 5. `label_parts(offset=N)` shifts a robot's
labels so instances can be told apart while keeping their class structure:

```python
for i, bot in enumerate(RobotSim.bots):
    bot.label_parts(offset=i * 10)     # per-robot instance segmentation
```

### Engine

Segmentation needs the object-index pass, which **EEVEE does not implement** —
the socket is not even offered on the Render Layers node. Requesting it switches
the scene to Cycles automatically.

That is not the compromise it sounds like. Measured here at 64x32 in background
mode:

| Engine | s/render |
| --- | --- |
| EEVEE, colour only | 3.118 |
| Workbench, colour only | 0.113 |
| Cycles @1 sample, colour only | 0.028 |
| Cycles @1 sample, **rgb + depth + segmentation** | 0.032 |
| Cycles @16 samples, all three passes | 0.065 |

Cycles is ~100x faster than EEVEE here *and* supports every pass — EEVEE has no
GPU in a headless container and falls back to software GL, which it pays for on
every render. Adding two extra modalities costs 0.004s, which is the multi-pass
argument in one number. `configure_cycles(samples=N)` trades noise for quality
when the RGB is training data rather than a smoke test.

The compositor graph is built once and left dormant between captures, so an
ordinary `quick_render()` does not start writing pass files as a side effect.

## Lidar

```python
bot = Robot(cameras='front', passes=('rgb',))
lidar = bot.add_lidar(channels=16, h_resolution=math.radians(0.5),
                      v_fov=math.radians(30), range_max=50)

@RobotSim
def tick(dt):
    scan = bot.scan_lidars()[0]          # every tick: cheap
    if scan.sector(-0.3, 0.3) < 1.5:     # something close, dead ahead
        bot.drive.drive(0.0, 1.0)
    views = bot.sample_cameras()         # rarely: expensive
```

A scan holds one range per beam in metres, plus the class label of whatever each
beam hit. Misses are `inf`, following the ROS `LaserScan` convention — a `0.0`
default would read as an obstacle touching the sensor, which is the most
dangerous possible way to be wrong.

```python
scan.ranges          # metres, beam order, inf for no return
scan.labels          # pass_index of the object each beam hit
scan.min_range       # closest return anywhere
scan.nearest()       # (range, azimuth, channel)
scan.sector(a, b)    # closest return in an angular sector, wraps through +/-pi
scan.points()        # hit positions, sensor frame (world=True for scene coords)
scan.to_mesh()       # point cloud as a Blender mesh, for looking at it
```

Defaults describe a cheap single-plane scanner: 360 degrees, one degree apart,
one channel, 50m. `channels > 1` spreads beams over `v_fov`, the multi-plane
arrangement of a spinning unit.

Lidar defaults to `interval=1` — **every tick**, unlike the cameras at 30. That
asymmetry is the whole point: the base reacts on the fast, cheap senses and only
peeks at vision when it needs to refresh its picture of the world.

### Why ray casting rather than a depth render

The depth pass is right there and looks like it should work. It does not, for
three reasons:

- Blender's Z pass is **planar depth**, not range. A flat wall reads the same
  value straight ahead and at the edge of frame, where the true slant range is
  `d/cos(theta)` — at 45 degrees that is 41% further. `test_lidar` asserts the
  real values (12.400, 12.837, 14.318, 17.536 across a sweep where a depth
  render would report a constant 12.400).
- A camera is a pinhole projection, so 360 degrees needs several renders
  stitched, and pixel columns are tan-spaced rather than evenly spaced in angle,
  so every beam needs resampling.
- Ray casting has neither problem, and the cast returns the object hit, so
  per-beam semantic labels come free.

It is also fast: **~3.7us per ray**, so a 360-beam scan costs 1.6ms and a
16-channel 11520-ray scan 43ms. Cost scales with beams, not with scene
resolution.

### Self-filtering

A lidar on a mast sees the robot's own arm and hubs. `add_lidar(self_filter=True)`
(the default) makes beams pass *through* the robot's own parts and carry on to
whatever is behind them, rather than being discarded — masking a beam entirely
would punch a permanent blind sector into every scan instead of merely occluding
what the mast hides. Each solid crossed costs two hits, near face and far face,
so the step-over budget is sized with headroom.

## Contact

Kinematic contact, built on the same ray caster the lidar uses. Not a physics
engine — still no mass, no momentum, no forces. What it adds is the part of
physics that matters for driving a camera around a world: the robot sits on the
ground instead of hovering, stops when it drives into something instead of
passing through, and reports the gap between what the wheels did and what the
base did.

```python
bot.enable_contact(level=True)     # ground following, collision, terrain pitch

bot.drive.slip                     # 0 running free, 1 held against a wall
bot.drive.last_contact.blocked     # what happened on the last step
bot.drive.last_contact.object      # what it hit
```

| Behaviour | What it does |
| --- | --- |
| ground following | rides `ride_height` above the surface under the wheels |
| collision | stops with the body's leading edge at the obstacle |
| sliding | a shallow approach skates along a wall instead of stopping dead |
| climbing | ramps and kerbs are driven onto; walls are not |
| levelling | pitches and rolls the base onto the slope under it |
| slip | commanded travel versus actual travel |

Contact points default to the wheels, so ground following and levelling agree
with where the robot actually touches. Cost is a handful of ray casts, about
330us per step for a four-wheel robot — cheaper than a lidar scan.

### The seam

`drive.py` previously integrated a commanded velocity straight into a pose,
which left nowhere for the world to have an opinion. `step()` now builds the
pose it *wants* and hands it to a `ContactModel`, which returns the pose it
actually gets **and the velocity it comes away with**:

```python
target, (v, omega), info = self.contact.resolve(self, start, target, (v, omega), dt)
```

Velocity is the half that makes the seam able to host a real solver. A pose-only
interface can stop a robot at a wall, but the robot resumes full speed the
instant the wall is gone, because nothing carried the impact forward. Returning
velocity lets contact say that hitting something *took your speed away*.

`ContactModel` is an interface with a no-op default, so `drive.py` stays free of
`bpy` and a robot without contact behaves exactly as before. `contact.RayContact`
is the ray-cast implementation; **an external physics engine slots in at this
same point**, which is what the drive interface was always shaped for.

## Inertia

The commanded twist is what the motors are being asked for; `drive.v` and
`drive.omega` are what the base is actually doing.

```python
bot = Robot(max_accel=1.0, max_yaw_accel=2.0)   # m/s^2 and rad/s^2
bot.drive.v                                      # actual, not commanded
```

Unset, the base reaches its commanded speed within one tick, which is what every
drive did before — so nothing changes unless it is asked for. Set, the actual
velocity lags the command at a bounded rate. That is momentum without ever
modelling a force: commanding a stop from 2 m/s at 1 m/s² coasts `v²/2a` = 2.0m
rather than halting instantly.

There is still no mass, no traction and no real dynamics. What there is now is a
base that cannot change speed instantly and a collision that costs it something,
which covers most of what a control policy needs to learn.

### Slip

Slip is the disagreement between the wheels and the base, scaled by whichever is
moving faster:

| Situation | Slip |
| --- | --- |
| up to speed, running free | 0 |
| spinning up under acceleration | high, falling to 0 |
| held against a wall | 1 |
| brakes locked, base still sliding | 1 |
| wheels driving forward, body still moving backward | >1 |

The last row is not a bug: when the wheels and the body disagree about
*direction*, the difference exceeds either one, and reporting 2.0 for a wheel
driving forward under a body sliding backward is more useful than clamping it to
1 and hiding the reversal.

Written as a ratio rather than `1 - actual/commanded` so it still means something
when the command is zero: locked brakes with the base still moving is *total*
slip, not undefined, and the simpler form silently reported it as none.

### Sharp edges worth knowing

A downward probe **cannot tell a floor from a ceiling**. Driving under an
overhanging ramp, the moment its underside comes within probe range the probe
reports it as ground. Ground is therefore rejected if it rises faster than
`max_climb` in one step — which is also why initial placement uses `snap()`,
unclamped, so a robot dropped in from any height reaches the surface in one go.

A robot at rest has its **wheel bottoms exactly coincident with the ground**.
Stepping a probe past the (ignored) wheel lands the ray a hair inside the ground,
where the next hit is the ground's *underside* — one slab-thickness too low,
every step, so the robot sinks. `cast_ignoring()` detects the back-face and
reports the entry point instead.

`max_climb` and `max_ground_rise` are deliberately separate settings. They
answer different questions — how tall a step can be driven onto, versus how fast
the ground may rise before it is really a ceiling — and a robot built to climb
tall steps would otherwise also start snapping onto overhangs.

Where contact points **straddle a discontinuity** — the lip where a ramp meets
its platform — the height difference across the robot is a cliff, not a slope,
and the raw levelling angle approaches vertical. `max_tilt` clamps it so one bad
probe cannot flip the robot.

## Firmware

robotsim's control loops are Python calling `drive.drive(v, omega)`. That is the
right level for working on the robot and the wrong level for asking whether the
C that will actually ship does the same thing. `firmware.py` closes that gap: the
application is compiled by [crust](https://github.com/brentharts/crust)'s
`hostsim`, runs as real machine code on its own thread, and reads and writes a
plant that is this simulator.

```python
bot = Robot()
fw = bot.attach_firmware('boards/drive_node.c', target=4000)   # counts
bot.step(dt)          # firmware and plant advance together

fw.board.motor_duty   # what the firmware is commanding
fw.board.encoder      # what it is being told
fw.board.console      # what it printed
```

`.c` builds directly; `.cpp` is lowered to plain C first by crust's C++ subset
front end, which **refuses** what it cannot lower rather than guessing — so a
rejection is a real answer about the source, not a gap in the toolchain.
`boards/drive_node.c` and `boards/pid_node.cpp` are the same position loop
written both ways.

### Optional, and checked

Clone crust (and optionally armulator) *beside* robotsim:

```
parent/
  robotsim/
  crust/
  armulator/
```

Nothing is imported unless they are there. `firmware.available()` reports
whether the path can run and `why_unavailable()` says what is missing, so
`make test_firmware` skips with an actionable message rather than failing on a
machine without the checkouts.

### Only hostsim can be in the loop

crust ships two ways to run an image and is emphatic that neither replaces the
other:

| | armulator | hostsim |
| --- | --- | --- |
| Executes AArch64 | yes | **no** |
| Speed | ~17k instructions/s | ~4000x faster |
| Answers | "does this image boot" | "does this system behave" |
| MMU, exception levels, registers | yes | no |
| numpy, matplotlib, sockets, CUDA | no | yes |

armulator is roughly 80,000x slower than real time, so a fifteen-second robotsim
run — 900 ticks — would take on the order of a fortnight. It belongs offline,
gating an image before it is trusted, which is why `firmware.armulator_root()`
exists but nothing steps it per tick.

The seam is also **values, not registers**: `sim_motor_write(duty)`, not a PWM
duty register. Firmware that programs a PCA9685 incorrectly works perfectly
here. That question belongs to armulator too.

### The encoder measures the wheels

`source_mode` decides what the firmware is told, and the default is the honest
one:

- **`'wheel'`** (default) — the wheels' own rotation, which is what a shaft
  encoder reads.
- **`'body'`** — ground-truth distance travelled. Perfect odometry no hardware
  has, useful for isolating a control bug from a sensing one.

The difference is not academic. Running the same firmware to the same 4000-count
setpoint:

| Scenario | Ground truth | Firmware believes |
| --- | --- | --- |
| `'body'` | 4.001 m | 4.001 m |
| `'wheel'` | 3.096 m | 4.006 m |
| driven into a wall | 1.153 m | 4.006 m |
| `fault_encoder_stuck()` | **28.95 m** (runaway) | 0.000 m |

The wheels reach commanded speed before the body does, and keep turning when the
robot is held, so dead reckoning drifts by the accumulated slip — an error a
Python control loop never has to face. Fault injection reaches further:
`fault_encoder_stuck` is a sensor that keeps reporting a plausible unchanging
value while the shaft turns, and the controller responds by commanding full duty
forever.

### Time

hostsim advances only when told to, which is what makes runs repeatable
regardless of host load — and it is why the board and the plant can share one
clock. `Board.step(dt)` takes seconds and carries the fractional counter ticks
between steps: `dt` of 1/60 s against a 19.2 MHz counter is a whole number of
ticks, but not every `dt` is, and truncating each step would drift the board's
clock away from the plant's silently and forever.

### Several boards

A robot can carry more than one MCU. `attach_firmware()` wires a board to the
base; `attach_board()` adds one that thinks and talks but drives nothing — a
planner, a vision node, an arm controller.

```python
drive = bot.attach_firmware('boards/drive_link_node.c', name='drive-mcu')
nav   = bot.attach_board('boards/nav_node.c', name='nav-mcu')
```

They talk over `bot.network`, a bus created on first use. Routing is crust's
`Fleet`, used as designed: `deliver()` needs only participants with a `name`,
`link_pop_all()` and `link_push()`. What robotsim does *not* use is
`Fleet.step()` — the clock stays here, because boards must advance in step with
the plant rather than on their own schedule.

`boards/nav_node.c` and `boards/drive_link_node.c` are that split: the planner
issues `T<counts>` setpoints and the base answers with `P<counts>` telemetry.

### One-step latency, and why

Delivery happens once every board has reached the same virtual time, so a
message sent during a tick arrives at the start of the next one. That latency is
deliberate — roughly what a real link costs — and it stops results depending on
the order boards happen to be listed in. `test_fleet` asserts it directly: nav
sends on tick 0, the drive board acts on tick 1.

Routing defaults to broadcast. `firmware.point_to_point(name)` narrows it, and
anything that reaches no recipient lands in `network.undelivered` rather than
disappearing.

### Across robots

```python
net = firmware.connect(lead, follower)     # one bus, two robots
RobotSim.networks.append(net)              # delivered after every robot steps
```

A shared bus has **no owner**, so no single robot's `step()` delivers it —
delivering from inside one robot's step would route messages before the others
had caught up, which is the same-virtual-time invariant the one-step latency
exists to preserve. `RobotSim.update()` delivers registered networks after every
robot has stepped; drive robots by hand and you call `net.deliver()` yourself.

### Grants are loop periods, not ticks

The least obvious thing in this integration. hostsim's `timer_count()` consumes
the **whole** grant in one call — it sets `now` to `deadline` — so a firmware
delay loop exits immediately however much time it was given, and the board
executes exactly **one loop iteration per grant**.

Handing a board a whole robotsim tick would therefore run a 1 kHz control loop
at the tick rate of 60 Hz, while the firmware's own arithmetic still believed it
was running at 1 kHz. Nothing would report an error; the loop would just be
sixteen times slow. So `Board.step(dt)` grants time in whole loop periods
(`loop_hz`, default 1000) and carries the remainder, which means the board trails
the plant by less than one period and never drifts.

```python
bot.attach_board('boards/vision_node.c', loop_hz=50)   # a slower loop
```

`test_fleet` pins both halves: 20,000 grants over 20 simulated seconds at 1 kHz,
and the planner's four-second legs actually firing five times.

### Link faults

```python
drive.board.fault_link_down(True)       # everything it sends is lost
drive.board.fault_link_drop_every(2)    # every other message
```

`link_send` returns a status because it can fail, and firmware that ignores it
loses messages exactly as it would on a real link. `nav_node.c` notices the
silence and says so; a simulation that cannot drop messages would never have
shown that.

## Datasets

The pipeline this simulator exists to feed needs aligned multi-modal samples: a
photorealistic image, the structural abstraction of the same frame, and the
semantic and depth maps that label it. `tools/generate_dataset.py` produces them.

```sh
make dataset                                  # 64 samples into /tmp/corpus
./tools/generate_dataset.py -- --samples 5000 --out /data/corpus
```

Each sample is one randomised scene rendered from one viewpoint four ways:

| Pass | Format | Contents |
| --- | --- | --- |
| `rgb` | PNG | the messy input a perception network must cope with |
| `lineart` | PNG | structure only — no texture, shadow or colour survives |
| `segmentation` | 32-bit EXR | integer class index per pixel |
| `depth` | 32-bit EXR | metres |

### Generating in parallel

```sh
./tools/generate_corpus.py --samples 5000 --out /data/corpus --workers 8 --prune
```

Each sample is an independent scene built from its own seed, so workers are
separate processes with nothing shared: worker `k` takes the samples where
`index % workers == k` and writes its own manifest, which are merged and sorted
by index afterwards. Nothing appends to a shared file, so there is no
interleaving to get subtly wrong under load.

Scene *content* depends on the index alone, so geometry, materials, lighting and
viewpoint are identical whatever the worker count -- measured bit-identical for
the photorealistic, depth and segmentation passes. **The line pass is the
exception:** Blender's stroke renderer carries state between renders within a
process, so changing how samples are divided shifts a few strokes by up to one
pixel (mean 0.5/255). It is bit-identical on rerun at a *fixed* worker count, so
reproducing a corpus exactly needs the seed **and** the worker count -- both are
recorded in `dataset.json`.

`--prune` drops samples that fail verification instead of failing the run. On a
480-sample corpus one sample came out featureless (mean 215.9, standard
deviation 0.44 -- a viewpoint that happened to face empty sky), which is roughly
what to expect at that rate. Pruned samples keep their files on disk so they can
still be looked at.

### Reproducibility

Every choice comes from a seeded generator owned by `randomize.py`, never from
global random state, and each sample's seed is stored in its manifest entry. A
sample that looks wrong during training can be regenerated on its own, without
rerunning the corpus. `Randomizer(seed).scene()` twice gives identical layouts;
`test_dataset` asserts it.

One ordering trap is worth knowing: `scene()` clears everything the randomizer
previously created, **lights included**. Create the scene first, then the
lights.

### The manifest is the dataset

`manifest.jsonl` carries one line per sample: its files, its seed, the camera,
and the label map in force when it was written. That last item is what turns the
ObjectID pass from an image into a semantic map — pixel value 7 means nothing
alone, and means `obstacle` only because the manifest says so *for that sample*.
Storing it per sample rather than once per corpus means a corpus whose labelling
changed halfway through is still readable instead of silently mislabelled.

JSON Lines rather than one document, because a run that dies at sample 40,000
should leave 40,000 usable samples rather than an unterminated array.

### Verification looks at pixels

```python
problems = ds.verify()    # [] when the corpus is sound
```

Alignment is the premise of the whole pipeline, and modalities that disagree on
resolution train a network to a systematic offset it can never recover from. So
`verify()` re-opens what was written and checks sizes agree — and checks the RGB
has tonal range, because **an unlit render is the corruption that looks like
success**: the file exists, the resolution matches, the manifest is complete, and
every pixel is black.

### Engines

The passes do not all come from one renderer. Depth and object index are
geometric and only Cycles exposes the object-index pass. Line art uses a flat
emission override, so it needs no lighting. The photorealistic pass is the one
that actually needs lights to work — and on the Blender packaged with this
container, **no light type illuminates a diffuse surface under Cycles**, so RGB
comes out black while depth and segmentation are perfectly correct. That is a
stripped build rather than a scene problem, but it is exactly the failure that
produces a large, well-formed, useless corpus.

So `--rgb-engine` defaults to EEVEE and `--rgb-engine cycles` restores the
single-engine path on a full build. The exposure check exists because of this.

## Stage 1: perception

The perception half of the chained architecture. It takes the messy
photorealistic frame and emits the structural abstraction a control policy is
trained on, so the policy never sees a texture, a shadow or a specular highlight
and cannot be confused by one.

```sh
./tools/generate_dataset.py -- --samples 500 --out /tmp/corpus
./tools/train_perception.py --corpus /tmp/corpus --epochs 60
```

`perception.py` needs no `bpy`, so training runs under plain Python and can go
somewhere with more cores than the machine that generated the corpus.

### The trap this is built around

Line art is about 99% white. **A network that outputs a blank page scores 0.99
accuracy and has learned nothing.** Two things follow, and they are the substance
of the module rather than details of it:

- the loss weights ink pixels far above background, so blankness is not the
  cheapest way down;
- the reported metric is F1 over ink pixels, never accuracy, and every score is
  printed beside what the blank page achieves on the same data.

Measured on a 479-sample corpus at 64x48, 48 epochs, on 96 validation samples:

| | F1 | precision | recall | accuracy |
| --- | --- | --- | --- | --- |
| blank page | 0.000 | — | 0.000 | 0.988 |
| trained, strict | 0.308 | 0.196 | 0.720 | 0.970 |
| trained, within 1px | **0.641** | 0.497 | 0.905 | — |
| trained, within 2px | 0.734 | 0.604 | 0.935 | — |

The trained model's *accuracy is worse than the blank page's* while its F1 goes
from nothing to a third. That row is the argument for the metric choice, in one
line.

### Strict and tolerant scores

Strokes are one pixel wide, so a prediction that traces a contour perfectly but
one pixel to the left scores **zero** on both precision and recall for that
contour. The strict number is therefore measuring localisation as much as
detection. Allowing a small matching tolerance -- how boundary detection is
normally scored -- separates the two, and the jump from 0.308 to 0.641 says the
network is finding the edges and placing them within a pixel.

Both are always reported together. A tolerant score is easy to quote without the
qualifier, and `test_perception` pins the two ways it could be abused: a
one-pixel-shifted perfect trace must go from 0.0 to 1.0, and predicting ink
everywhere must *not* score well (recall 1.000, precision 0.164, F1 0.282).

### What the bottleneck is not

Three hypotheses were tested and two were wrong, which is worth recording so
they are not tried again:

- **More data.** Going from 140 to 479 samples did not raise the score. Train F1
  0.367 against val F1 0.305 -- a gap of 0.06 -- means the model cannot fit the
  training set either. That is underfitting, so the corpus was never the limit.
- **Receptive field.** Five 3x3 layers see 11 pixels, which sounded too local to
  judge an object boundary. Dilations of 1,2,4,8 widen that to 63 pixels and
  scored *worse* at matched epochs (0.256 vs 0.279 at 12 epochs, 0.307 vs 0.326
  at 24). `--dilations` remains available; it is not the answer here.
- **Threshold.** Tuned on training data it gave 0.325 on validation against
  0.341 at the default, so the default stands. Choosing it on the validation set
  would have "improved" the number, which is how a tuned threshold becomes an
  inflated score.

What remains: capacity (24 channels is small), and the loss -- an exact-pixel
target punishes a one-pixel miss as hard as a total one, and the tolerance
analysis above says that is exactly the error the model is making.

### Why numpy rather than torch

Torch is the right tool and the module is shaped so it can be swapped in --
`LineArtNet` is a plain stack of layers behind `forward`/`backward`, and nothing
above it assumes how the gradients were produced.

It is not used here because the PyPI Linux wheel links CUDA libraries even for
CPU-only use: `libtorch_global_deps.so` needs libcublas at import, `--no-deps`
therefore cannot work, and the dependency chain measures over four gigabytes
unpacked. Rather than ship code that cannot be run and therefore cannot be
trusted, the reference implementation is numpy -- slower, and small enough to
verify against a numerical gradient check.

That check is in `test_perception`, and it runs in float64 deliberately: a
numerical derivative is a difference of two nearly equal numbers, and in float32
the cancellation swamps the result. The check then fails on arithmetic rather
than on a wrong gradient, which is a confusing hour to spend.

## Telemetry

A render shows where the robot ended up. It does not show why — the duty the
firmware was commanding, the slip between wheels and ground, the moment contact
took the velocity away, the range that triggered the turn.

```python
tel = Telemetry(name='run')
tel.watch_robot(bot, prefix='base')
tel.watch_board(drive.board)
tel.watch_lidar(lidar, sectors={'ahead': (-0.3, 0.3)})
tel.watch('battery', lambda: pack.volts, group='power', unit='V')

@RobotSim
def tick(dt):
    tel.sample(RobotSim.time)

tel.plot('/tmp/run.png')
telemetry.compose('/tmp/render.png', '/tmp/run.png', '/tmp/frame.png')
```

Sampling is pull-based: the recorder holds a callable per channel and reads them
when told, so the simulator needs no knowledge that telemetry exists and a custom
signal is one lambda rather than a plumbing change. Channels sharing a `group`
share a subplot — duty against encoder is noise; duty against its own limits is a
story — and every subplot shares one time axis, since the question a panel
answers is "what else was happening when this happened".

`mark(label)` draws a vertical line across every plot, which is how a setpoint
change or a collision gets tied to the wiggle it caused. Boards passed to
`watch_board()` get their console printed beneath the plots: the plot says the
duty saturated, the console says which branch decided that.

matplotlib is optional, like crust. Recording never needs it and `to_csv()` gets
the data out regardless; only `plot()` does, and `available()` reports whether it
can.

Two details that matter more than they look. Infinite readings — a lidar that saw
nothing — are recorded as the sensor's own maximum, because rescaling an axis to
infinity hides every real reading on it. And a signal that barely moves gets a
floor on its axis span (`min_span`), or a robot sitting flat at 0.15 m draws a
height axis spanning 1e-4 and appears to be bouncing.

## Base motion

Kinematic models — they integrate a velocity command into a pose and spin the
wheel meshes to match. No mass and no traction; contact and slip are supplied by
a `ContactModel` rather than by the drive itself.

```python
bot.drive.drive(v, omega)              # differential: body-frame command
bot.drive.set_wheel_speeds(l, r)       # differential: wheel-space command
car.drive.drive(speed, steer)          # ackermann: rear-drive, front-steer
```

`AckermannDrive` gives the front wheels true Ackermann angles — the inner wheel
turns sharper than the outer — so the rendered geometry is correct.

## Testing

```bash
make test              # camera render smoke test
make test_anim         # multi-robot animated render, writes a GIF
make test_joints       # joint read/write, limits, IK round-trip
make test_drive        # drive models, wheel roll, sim clock
make test_record       # timeline binding, scrub reproduces sim state
make test_arm_record   # arm recording in both IK and FK modes, plus export
make test_rig          # wheel layouts, custom wheel and arm placement
make test_sensors      # multi-pass capture, metric depth, segmentation labels
make test_lidar        # beam geometry, radial range, self-filtering, labels
make test_contact      # ground following, blocking, sliding, ramps, slip
make test_firmware     # real C/C++ firmware in the loop (skips without crust)
make test_fleet        # several MCUs per robot, messaging, routing, link faults
make test_telemetry    # channel recording, axis scaling, panels (skips without matplotlib)
make test_dataset      # randomisation determinism, line art, corpus verification
make dataset           # generate a 64-sample corpus into /tmp/corpus
make test_perception   # gradient check, the blank-page baseline, learning
make train             # train Stage 1 on /tmp/corpus
make corpus            # parallel corpus generation across CPUs
make test_all          # everything
```

The suite runs in about a minute. Most of the remaining time is the handful of
EEVEE renders kept deliberately in `test` to cover the production render path;
everything else runs on Workbench.

Tests assert against physical invariants rather than golden values — a closed
circle returns to its start, scrubbing the timeline reproduces the pose the sim
had, a commanded joint reaches the angle it was given.

## Status

Working today:

- Procedural robot construction: base, wheels, hubs, camera mast, multiple arms
- Joint-space and task-space arm control with limits and IK management
- Differential / skid-steer and Ackermann base motion on a real timestep
- Arbitrary wheel counts, laid out automatically or placed by hand
- Multiple arms on custom mounts, including onto a hub, and added at runtime
- Configurable camera sets and a capture interval, so vision runs on its own
  slower clock than the control loop
- Multi-pass sensor capture: RGB, metric depth and semantic/instance
  segmentation, all from a single render per camera
- Ray-cast lidar: multi-channel, 360 degrees, true radial range, per-beam
  labels, fast enough to run every tick
- Kinematic contact: ground following, terrain levelling, collision with
  sliding, and slip reporting, behind a seam an external solver can replace
- Inertia: acceleration limits, and collisions that consume velocity rather
  than only correcting position
- Firmware in the loop: real C/C++ compiled by crust's hostsim, driving the
  simulated plant through a shaft encoder, with fault injection
- Multi-board fleets: several MCUs per robot, board-to-board and robot-to-robot
  messaging with one-step latency, routing and link faults
- Telemetry panels: any signal recorded per tick, drawn with matplotlib and
  composed with the Blender render
- Multi-camera rendering per robot, per frame
- Timeline recording, scrubbing, and standard Blender animation rendering
- ROS2-style joint trajectory export from any recording

Not yet built:

- Dynamics: no mass, traction or forces. Momentum is an acceleration limit,
  not an integrated one.
- External physics engine hookup (it slots into `drive.ContactModel`)
- Proximity/contact sensors (RGB, depth, segmentation and lidar are done)
- The armulator path: register-level driver verification, offline
- The PyTorch training loop and Jetson deployment path

---

