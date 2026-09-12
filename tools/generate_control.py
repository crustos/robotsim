#!/usr/bin/env ../headless.py
"""
Generate a Stage 2 control corpus: abstraction in, expert command out.

    ./tools/generate_control.py -- --episodes 12 --out /tmp/control

Each episode drops the robot in a randomised scene with obstacles and a visible
goal marker, drives it with the privileged expert in `control.py`, and records
at every step the abstraction the robot's own camera saw -- line art plus a
semantic map -- beside the command the expert issued. The photorealistic pass is
never rendered: Stage 2 is defined not to see it, so spending a second render on
it would only invite training against it by accident.

Output is a single `.npz` per shard holding observations, actions, poses and
goals. Not the JSON-manifest format the perception corpus uses, because a
control corpus is thousands of small arrays rather than hundreds of image files,
and one array file loads in a single read.

Sharding exists for the same reason it does in `generate_dataset.py`: episodes
are independent, so the work divides cleanly. Each shard seeds itself from its
own episode indices, so a corpus is identical however it was divided.
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
    """
    Ground, obstacles and a goal marker, labelled for the semantic map.

    The goal is a tall thin post rather than a flat patch: it has to be visible
    from across the scene and from a camera mounted low on a robot, and a mark
    painted on the floor is hidden by the first obstacle between it and the
    lens. A policy that cannot see the goal learns the average heading of the
    training set, which reads as progress and is not.
    """
    obstacles = rnd.scene(obstacles=(n_obstacles, n_obstacles),
                          label=C.OBSTACLE_CLASS)
    rnd.materials(obstacles)

    ## Somewhere clear of the origin, where the robot starts.
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
    ap.add_argument('--episodes', type=int, default=12)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--out', default='/tmp/control')
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--height', type=int, default=48)
    ap.add_argument('--extent', type=float, default=8.0)
    ap.add_argument('--obstacles', type=int, default=3)
    ap.add_argument('--dt', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--episode-offset', type=int, default=0,
                    help='added to both the episode seed and the recorded id, '
                         'so a second batch extends a corpus instead of '
                         'silently duplicating it. Ids must stay unique: the '
                         'train/val split is by episode, and two batches '
                         'sharing an id put the same rollout on both sides.')
    ap.add_argument('--physics', action='store_true',
                    help='use the MuJoCo contact backend, so the recorded '
                         'poses are what a robot with mass would reach rather '
                         'than what the command asked for')
    args = ap.parse_args(argv)

    resolution = (args.width, args.height)
    work = os.path.join('/tmp', 'robotsim-control-work')
    os.makedirs(work, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

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

    episodes = [e for e in range(args.episodes) if e % args.shards == args.shard]
    observations, actions, poses, goals, episode_ids = [], [], [], [], []

    for episode in episodes:
        episode = episode + args.episode_offset
        rnd.reseed(args.seed * 7919 + episode)
        post, goal, circles = build_scene(rnd, args.extent, args.obstacles)

        rnd.lighting(count=(1, 3), energy=(3.0, 12.0))
        rnd.world()
        bot.root.location = (0.0, 0.0, 0.15)
        bot.root.rotation_euler = (0.0, 0.0, rnd.rng.uniform(-0.6, 0.6))
        bpy.context.view_layer.update()

        contact = None
        if args.physics:
            ## Dynamic contact: the pose recorded is the one the robot actually
            ## reached. Without it the expert's command and the resulting pose
            ## agree by construction, and the corpus never contains a frame in
            ## which the robot is somewhere the command did not put it.
            contact = bot.enable_contact(backend='mujoco', mu=1.0)

        for step in range(args.steps):
            pose = (bot.root.location.x, bot.root.location.y,
                    bot.root.location.z, bot.root.rotation_euler.z)
            v, omega = expert.act(pose, goal, circles)

            capture = bot.capture(frame=step, resolution=resolution)['front']
            seg = read_pass(capture['segmentation'])
            ink_path = line.render(camera,
                                   os.path.join(work, 'ep%04d_%03d.png' % (episode, step)),
                                   resolution=resolution)
            ink = read_grey(ink_path)

            observations.append(C.observation(ink, seg))
            actions.append(C.encode_action((v, omega)))
            poses.append([pose[0], pose[1], pose[3]])
            goals.append([goal[0], goal[1]])
            episode_ids.append(episode)

            advance(bot, contact, v, omega, args.dt)

            if math.hypot(goal[0] - bot.root.location.x,
                          goal[1] - bot.root.location.y) < expert.arrive:
                break

        bpy.data.objects.remove(post, do_unlink=True)

    ## The offset is part of the filename, not just the episode ids. Two
    ## batches written to the same directory with different offsets are
    ## different data, and naming them both `control.000.npz` silently destroys
    ## the first -- which is a corpus, not a cache.
    suffix = ''
    if args.episode_offset:
        suffix += '.b%d' % args.episode_offset
    if args.shards > 1:
        suffix += '.%03d' % args.shard
    path = os.path.join(args.out, 'control%s.npz' % suffix)
    np.savez_compressed(path,
                        observations=np.asarray(observations, dtype=np.float32),
                        actions=np.asarray(actions, dtype=np.float32),
                        poses=np.asarray(poses, dtype=np.float32),
                        goals=np.asarray(goals, dtype=np.float32),
                        episodes=np.asarray(episode_ids, dtype=np.int32))
    print('wrote %s: %d frames from %d episodes'
          % (path, len(observations), len(episodes)))
    return 0


def advance(bot, contact, v, omega, dt):
    """
    Move the robot one tick, through physics when it is enabled.

    Falls back to integrating the twist directly, which is what the kinematic
    backend does anyway. Kept explicit rather than always going through the
    drive model so that a corpus can be generated without contact at all.
    """
    if contact is not None:
        pose = (bot.root.location.x, bot.root.location.y,
                bot.root.location.z, bot.root.rotation_euler.z)
        pose, _velocity, _info = contact.resolve(None, pose, pose, (v, omega), dt)
        bot.root.location = (pose[0], pose[1], pose[2])
        bot.root.rotation_euler = (0.0, 0.0, pose[3])
    else:
        yaw = bot.root.rotation_euler.z + omega * dt
        bot.root.rotation_euler = (0.0, 0.0, yaw)
        bot.root.location = (bot.root.location.x - math.sin(yaw) * v * dt,
                             bot.root.location.y + math.cos(yaw) * v * dt,
                             bot.root.location.z)
    bpy.context.view_layer.update()


def read_pass(path):
    """One channel of an EXR pass, top-down, as a 2-D array."""
    image = bpy.data.images.load(path)
    width, height = image.size
    pixels = np.array(image.pixels[:], dtype=np.float32)
    bpy.data.images.remove(image)
    return np.flipud(pixels.reshape(height, width, 4)[:, :, 0])


def read_grey(path):
    from PIL import Image
    with Image.open(path) as img:
        return np.asarray(img.convert('L'), dtype=np.float32) / 255.0


## No __main__ guard: robotsim execs this script into its own globals, so
## __name__ is never '__main__' here, and `bpy`, `Robot`, `Randomizer`,
## `LineArt` and `create_cube` arrive from those globals rather than by import.
_argv = sys.argv[len(sys.argv) - 1 - sys.argv[::-1].index('--') + 1:] if '--' in sys.argv else []
_status = main(_argv)
if _status:
    raise SystemExit(_status)
