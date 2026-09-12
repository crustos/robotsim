#!/usr/bin/env ../headless.py
"""
Drive the chain and see whether it gets there.

    ./tools/evaluate_control.py -- --policy /tmp/control_net.npz --episodes 8

Frame-level regression error is not a control result. A cloned policy can match
the expert on every held-out frame and still fail the moment it drives, because
its own small errors carry it into states the expert never visited and the
training set therefore never contained. This closes the loop: the policy is put
in the driving seat, renders its own observations from wherever it has ended up,
and is scored on whether it reached the goal.

Three drivers are run over the *same* scenes, which is the only way the numbers
compare:

  expert    privileged, sees true poses. The ceiling.
  policy    sees only the rendered abstraction. The thing under test.
  forward   drives straight at constant speed. The floor -- and not a silly
            one, because a goal placed ahead of the robot is sometimes reached
            by accident, and a policy that cannot beat this has learned nothing
            about steering even if its frame error looks respectable.

The scene construction is deliberately duplicated from `generate_control.py`
rather than imported: robotsim execs its tools into its own globals, so one tool
cannot import another, and the alternative is a shared module that only works
inside Blender.
"""

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import control as C                                          # noqa: E402


def build_scene(rnd, extent, n_obstacles):
    obstacles = rnd.scene(obstacles=(n_obstacles, n_obstacles),
                          label=C.OBSTACLE_CLASS)
    rnd.materials(obstacles)
    while True:
        gx = rnd.rng.uniform(-extent * 0.35, extent * 0.35)
        gy = rnd.rng.uniform(extent * 0.15, extent * 0.4)
        if math.hypot(gx, gy) > extent * 0.2:
            break
    post = create_cube('GOAL', size=(0.35, 0.35, 1.6), location=(gx, gy, 0.8))
    post.pass_index = C.GOAL_CLASS
    material = bpy.data.materials.new('GOAL.MAT')
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.85, 0.15, 0.1, 1.0)
    post.data.materials.append(material)
    circles = []
    for obj in obstacles:
        centre = obj.matrix_world.translation
        radius = max(obj.dimensions[0], obj.dimensions[1]) * 0.5
        circles.append((centre.x, centre.y, radius))
    return post, (gx, gy), circles


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--policy', default='/tmp/control_net.npz')
    ap.add_argument('--episodes', type=int, default=8)
    ap.add_argument('--steps', type=int, default=16)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--height', type=int, default=48)
    ap.add_argument('--extent', type=float, default=8.0)
    ap.add_argument('--obstacles', type=int, default=3)
    ap.add_argument('--dt', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=500,
                    help='deliberately different from the generator default, '
                         'so evaluation scenes are not training scenes')
    ap.add_argument('--drivers', default='expert,policy,forward')
    ap.add_argument('--out', default='/tmp/control_eval.json')
    args = ap.parse_args(argv)

    resolution = (args.width, args.height)
    work = os.path.join('/tmp', 'robotsim-eval-work')
    os.makedirs(work, exist_ok=True)

    ground = create_cube('GROUND', size=(400, 400, 0.2), location=(0, 0, -0.1))
    ground.pass_index = 1
    ground_mat = bpy.data.materials.new('GROUND.MAT')
    ground_mat.use_nodes = True
    bsdf = ground_mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.35, 0.35, 0.37, 1.0)
    ground.data.materials.append(ground_mat)

    bot = Robot(cameras='front', arms=[], passes=('segmentation',), out_dir=work)
    camera = bot.cameras['front']
    rnd = Randomizer(seed=args.seed, extent=args.extent)
    line = LineArt(thickness=1.1)
    expert = C.Expert()

    policy = None
    if 'policy' in args.drivers:
        if not os.path.isfile(args.policy):
            raise SystemExit('no policy at %s; train one first' % args.policy)
        policy = C.ControlNet()
        policy.load(args.policy)

    drivers = [d.strip() for d in args.drivers.split(',') if d.strip()]
    results = {d: [] for d in drivers}

    for episode in range(args.episodes):
        for driver in drivers:
            ## Same seed per episode across drivers, so every driver faces the
            ## identical scene. Comparing drivers on different scenes measures
            ## the scenes.
            rnd.reseed(args.seed * 31 + episode)
            post, goal, circles = build_scene(rnd, args.extent, args.obstacles)
            rnd.lighting(count=(1, 3), energy=(3.0, 12.0))
            rnd.world()
            bot.root.location = (0.0, 0.0, 0.15)
            bot.root.rotation_euler = (0.0, 0.0, 0.0)
            bpy.context.view_layer.update()

            reached = False
            closest = float('inf')
            for step in range(args.steps):
                x, y = bot.root.location.x, bot.root.location.y
                yaw = bot.root.rotation_euler.z
                distance = math.hypot(goal[0] - x, goal[1] - y)
                closest = min(closest, distance)
                if distance < expert.arrive:
                    reached = True
                    break

                if driver == 'expert':
                    v, omega = expert.act((x, y, 0.0, yaw), goal, circles)
                elif driver == 'forward':
                    v, omega = (C.MAX_V, 0.0)
                else:
                    ## The policy renders its own observation from wherever it
                    ## has driven itself to. This is the part a frame-level
                    ## score cannot test.
                    capture = bot.capture(frame=step, resolution=resolution)['front']
                    seg = read_pass(capture['segmentation'])
                    ink = read_grey(line.render(
                        camera, os.path.join(work, 'e%03d_%03d.png' % (episode, step)),
                        resolution=resolution))
                    v, omega = policy.act(C.observation(ink, seg))

                advance(bot, v, omega, args.dt)

            x, y = bot.root.location.x, bot.root.location.y
            final = math.hypot(goal[0] - x, goal[1] - y)
            closest = min(closest, final)
            results[driver].append({'episode': episode, 'reached': reached,
                                    'final': final, 'closest': closest})
            bpy.data.objects.remove(post, do_unlink=True)

    print('\n%-8s %8s %14s %14s' % ('driver', 'reached', 'mean final', 'mean closest'))
    summary = {}
    for driver in drivers:
        rows = results[driver]
        rate = sum(1 for r in rows if r['reached']) / max(1, len(rows))
        finals = [r['final'] for r in rows]
        closes = [r['closest'] for r in rows]
        summary[driver] = {'reached': rate,
                           'mean_final': float(np.mean(finals)),
                           'mean_closest': float(np.mean(closes)),
                           'episodes': rows}
        print('%-8s %7.0f%% %13.2fm %13.2fm'
              % (driver, rate * 100, np.mean(finals), np.mean(closes)))

    import json
    with open(args.out, 'w') as fh:
        json.dump(summary, fh, indent=1)
    print('\n-> %s' % args.out)
    return 0


def advance(bot, v, omega, dt):
    yaw = bot.root.rotation_euler.z + omega * dt
    bot.root.rotation_euler = (0.0, 0.0, yaw)
    bot.root.location = (bot.root.location.x - math.sin(yaw) * v * dt,
                         bot.root.location.y + math.cos(yaw) * v * dt,
                         bot.root.location.z)
    bpy.context.view_layer.update()


def read_pass(path):
    image = bpy.data.images.load(path)
    width, height = image.size
    pixels = np.array(image.pixels[:], dtype=np.float32)
    bpy.data.images.remove(image)
    return np.flipud(pixels.reshape(height, width, 4)[:, :, 0])


def read_grey(path):
    from PIL import Image
    with Image.open(path) as img:
        return np.asarray(img.convert('L'), dtype=np.float32) / 255.0


_argv = sys.argv[len(sys.argv) - 1 - sys.argv[::-1].index('--') + 1:] if '--' in sys.argv else []
_status = main(_argv)
if _status:
    raise SystemExit(_status)
