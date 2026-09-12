#!/usr/bin/env python3
"""
The armulator bridge: robotsim's plant gated against register-level hardware models.

Unlike the rest of tests/, this one is plain Python rather than `#!../headless.py`.
That is the point rather than an omission -- the gate exists to run offline, before an
image is trusted, on a build machine that has no Blender. It exercises robotsim's real
`DifferentialDrive.step` through `HeadlessRoot`, so what is being gated is the shipping
integrator and not a stand-in for it.
"""

import math
import os
import sys

print('hello armulator bridge test...')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import armulator_bridge as bridge          # noqa: E402

## Optional dependency, same contract as firmware_test: reported, not raised.
if not bridge.available():
    print('SKIP armulator bridge test: %s' % bridge.why_unavailable())
    print('armulator bridge test OK (skipped)')
    raise SystemExit(0)

DT = 1.0 / 60.0


def close(a, b, tol=1e-3):
    return abs(a - b) < tol


## ------------------------------------------------------------- unit conversion
plant = bridge.MotorPlant(wheel_radius=0.1, gear_ratio=1.0, free_speed=5.0)

## A 0.1m wheel turning once covers its circumference.
assert close(plant.shaft_to_surface(1.0), 2 * math.pi * 0.1), plant.shaft_to_surface(1.0)
assert close(plant.surface_to_shaft(plant.shaft_to_surface(2.3)), 2.3)
print('max wheel speed: %.3f m/s' % plant.max_wheel_speed)

## Gearing down trades speed for the torque this model does not simulate, but the
## kinematics still have to be right.
geared = bridge.MotorPlant(wheel_radius=0.1, gear_ratio=20.0, free_speed=5.0)
assert close(geared.max_wheel_speed, plant.max_wheel_speed / 20.0)


## ---------------------------------------------------------------- the driver
## Writing a speed must reach the shaft as a duty and a direction.
plant.set_wheel_speeds(1.0, 1.0)
assert plant.left.bridge.drive > 0, plant.left.bridge.drive
assert plant.right.bridge.drive > 0, plant.right.bridge.drive

plant.set_wheel_speeds(-1.0, 1.0)
assert plant.left.bridge.drive < 0, 'left wheel did not reverse'
assert plant.right.bridge.drive > 0, 'right wheel should still be forward'

## M1 and M2 have their direction channels in opposite orders on the real HAT. A
## bridge that assumed a uniform pattern would drive one of these backwards, so a
## spin command must actually produce opposite signs.
assert plant.left.bridge.drive * plant.right.bridge.drive < 0, \
    'channel routing collapsed a spin into both wheels the same way'

## Brake is not coast, and the difference is a real driver bug.
plant.brake()
assert plant.left.bridge.braking, 'brake did not short the bridge'
assert not plant.left.bridge.mode == 'coast'
print('driver: direction, routing and brake all reach the bridges')


## ------------------------------------------------------------ straight line
straight = bridge.compare_drive([(3.0, 1.0, 0.0)], dt=DT)
print(straight.format())

## The register path lags: a real motor spins up rather than arriving at speed.
## So it must be behind, and by a bounded amount, not ahead and not wildly off.
assert straight.final_position_error > 0.0, \
    'no divergence at all -- the register path is probably not wired in'
## Gated on steady-state, not peak: the transient below is spin-up, and is correct.
assert straight.max_speed_error > 0.5, 'expected a visible spin-up transient'
assert straight.passes(speed=0.05, position=0.5), straight.format()

ref_y = straight.reference_pose[1]
plant_y = straight.plant_pose[1]
assert plant_y < ref_y, 'register path should lag, not lead (%.3f vs %.3f)' % (
    plant_y, ref_y)
print('straight line: register path lags by %.4f m over 3s' % (ref_y - plant_y))

## Both went essentially straight; spin-up is symmetric so heading holds.
assert close(straight.final_heading_error, 0.0, tol=1e-6), straight.final_heading_error


## ------------------------------------------------------------------ turning
turn = bridge.compare_drive([(3.0, 0.5, 0.6)], dt=DT)
print(turn.format())
assert turn.passes(speed=0.05, position=0.5, heading=math.radians(15)), turn.format()
print('turn: %.4f rad heading divergence' % turn.final_heading_error)


## ------------------------------------------------------------- stall deadband
## Below stall_drive the shaft does not turn at all, where robotsim's value seam
## happily crawls. This is the divergence most likely to surprise someone, so it
## is asserted rather than left to be discovered.
crawl = bridge.compare_drive([(2.0, 0.02, 0.0)], dt=DT)
crawl_ref = crawl.reference_pose[1]
crawl_plant = crawl.plant_pose[1]
print('crawl at 0.02 m/s: reference %.4f m, register %.6f m' % (crawl_ref, crawl_plant))
assert crawl_ref > 0.03, 'reference path should have crawled forward'
assert close(crawl_plant, 0.0, tol=1e-6), \
    'register path moved below the stall threshold: %.6f' % crawl_plant
print('stall deadband reproduced: the value seam crawls, the real motor does not')


## ---------------------------------------------------------------- lidar world
## A stub world with known geometry: a corridor two metres either side. Using a
## stub rather than the Blender caster is what separates a protocol bug from a
## ray-casting bug -- here the geometry is exact by construction.
def corridor(origin, bearing):
    """Distance to a wall at x = +/-2.0 from a point inside."""
    dx = math.cos(bearing)
    if abs(dx) < 1e-9:
        return None
    wall = 2.0 if dx > 0 else -2.0
    distance = (wall - origin[0]) / dx
    return distance if distance > 0 else None


world = bridge.SceneWorld(corridor, name='corridor')
assert close(world.range_at((0.0, 0.0), 0.0), 2.0)
assert close(world.range_at((0.5, 0.0), 0.0), 1.5)
assert close(world.range_at((0.5, 0.0), math.pi), 2.5)

try:
    bridge.SceneWorld('not callable')
except TypeError as error:
    print('SceneWorld rejects a non-callable cast: %s' % error)
else:
    raise AssertionError('SceneWorld accepted a non-callable cast')

report = bridge.compare_lidar(world, seconds=1.0, dt=DT, max_range=6.0)
print(report.format())

assert report.compared > 100, 'too few ranges compared: %d' % report.compared
## Every decoded range must match the world it came from to within the protocol's
## own resolution. Anything larger is the register or wire path corrupting it.
assert report.passes(), report.format()
print('lidar: %d ranges survived the full register and wire path intact'
      % report.compared)

## The naive difference is much larger than the distance quantisation, and that is
## the angular term. Asserting it is what stops someone "fixing" the bracket away.
assert report.max_raw_error > report.QUANTISATION * 4, (
    'expected angle quantisation to dominate, got %.6f m' % report.max_raw_error)
print('angle quantisation accounts for %.6f m of apparent error'
      % report.max_raw_error)


## ------------------------------------------------------------------ reporting
assert 'drive conformance' in straight.format()
assert 'lidar conformance' in report.format()
assert repr(straight).startswith('<DriveReport')
assert repr(report).startswith('<LidarReport')

print('armulator bridge test OK')
