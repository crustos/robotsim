#!../headless.py
"""
Generate an aligned multi-modal corpus.

    ./tools/generate_dataset.py -- --samples 200 --out /tmp/corpus

Each sample is one randomised scene rendered four ways from one viewpoint:

    rgb           photorealistic, the messy input
    lineart       structural, the abstraction a policy is trained on
    segmentation  integer class index per pixel
    depth         metres

The four are pixel-aligned by construction -- the camera does not move between
them -- which is the property the chained architecture depends on and which
`Dataset.verify()` re-checks after the fact.

ENGINES
-------
The passes do not all come from the same renderer, and the reason is worth
recording rather than hiding in a flag.

Depth and object index are geometric: the renderer resolves them while shading,
and only Cycles exposes the object-index pass at all. Line art is drawn with a
flat emission override, so it needs no lighting and Cycles renders it happily.
The photorealistic pass is the exception -- it is the one that actually needs
lights to work.

On the Blender packaged with this container, they do not: no light type
illuminates a diffuse surface under Cycles, so an RGB frame comes out black
while depth and segmentation are perfectly correct. That is an artefact of a
stripped build rather than of the scene, but it is precisely the failure that
would otherwise produce a large, well-formed, useless corpus. So the RGB pass
defaults to a different engine here, and `--rgb-engine cycles` restores the
single-engine path on a full build.

The exposure check in `Dataset.verify()` exists because of this: a corpus whose
lighting silently failed passes every structural check there is.
"""

import argparse
import math
import os
import sys
import time


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='/tmp/corpus')
    ap.add_argument('--samples', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--height', type=int, default=192)
    ap.add_argument('--cycles-samples', type=int, default=16)
    ap.add_argument('--bounces', type=int, default=4)
    ap.add_argument('--rgb-engine', default='eevee',
                    choices=('eevee', 'cycles', 'workbench'))
    ap.add_argument('--split', default='train')
    ap.add_argument('--obstacles', type=int, nargs=2, default=(3, 9))
    ap.add_argument('--extent', type=float, default=10.0)
    ap.add_argument('--no-lineart', action='store_true')
    ap.add_argument('--keep-going', action='store_true',
                    help='report bad samples instead of stopping at the first')
    ap.add_argument('--shard', type=int, default=0,
                    help='which slice of the corpus this process generates')
    ap.add_argument('--shards', type=int, default=1,
                    help='how many processes are generating in total')
    ap.add_argument('--no-verify', action='store_true',
                    help='skip verification (a shard cannot verify the whole)')
    ap.add_argument('--muble-scenes', default=None,
                    help='render MuBlE scenes instead of procedural ones: a '
                         'handoff file, a directory of them, or a raw MuBlE '
                         'scenes.json')
    ap.add_argument('--muble-root', default=None,
                    help='MuBlE checkout to resolve shapes and materials '
                         'against (default: whatever the handoff recorded)')
    args = ap.parse_args(argv)

    resolution = (args.width, args.height)
    work = os.path.join('/tmp', 'robotsim-gen-work')
    os.makedirs(work, exist_ok=True)

    ground = create_cube('GROUND', size=(400, 400, 0.2), location=(0, 0, -0.1))
    ground.pass_index = SEGMENT_CLASSES['ground']
    ## A material on the ground, because a diffuse surface with no material is
    ## one more thing that can render black for an uninteresting reason.
    ground_mat = bpy.data.materials.new('GROUND.MAT')
    ground_mat.use_nodes = True
    bsdf = ground_mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.35, 0.35, 0.37, 1.0)
    ground.data.materials.append(ground_mat)

    passes = ['rgb', 'depth', 'segmentation'] + ([] if args.no_lineart else ['lineart'])
    bot = Robot(cameras='front', arms=[], passes=('depth', 'segmentation'),
                out_dir=work)
    camera = bot.cameras['front']

    rnd = Randomizer(seed=args.seed, extent=args.extent)
    line = LineArt(thickness=1.1)

    ## Each shard owns a disjoint set of indices and its own manifest. Only
    ## shard 0 may clear the directory, and only because the launcher runs it
    ## before the others start -- a shard wiping the output while its siblings
    ## are writing into it is the obvious way to lose a corpus.
    sharded = args.shards > 1
    ds = Dataset(args.out, passes=tuple(passes), split=args.split,
                 overwrite=not sharded, note='robotsim procedural corpus',
                 manifest=('manifest.%03d.jsonl' % args.shard) if sharded else None)
    for token, index in SEGMENT_CLASSES.items():
        ds.label(token, index)
    ds.label('obstacle', OBSTACLE_LABEL)

    ## MuBlE scenes replace the procedural ones as the source of *geometry*.
    ## Lighting, world and viewpoint stay randomised on top, because the whole
    ## point of the corpus is appearance variation over fixed structure -- a
    ## fixed scene rendered identically every time teaches Stage 1 nothing.
    muble_scenes = load_muble(args.muble_scenes, args.muble_root) if args.muble_scenes else None
    if muble_scenes:
        print('using %d MuBlE scene(s) as geometry' % len(muble_scenes))
        ## The robot is a metre-scale vehicle and these are tabletop scenes; it
        ## would fill the frame. Hidden from render rather than deleted, because
        ## its camera is still what captures the sample.
        for part in bot.parts():
            part.hide_render = True

    indices = [i for i in range(args.samples) if i % args.shards == args.shard]
    print('generating %d samples (shard %d/%d) -> %s'
          % (len(indices), args.shard, args.shards, args.out))
    started = time.time()
    for done, i in enumerate(indices):
        seed = args.seed * 1000003 + i
        rnd.reseed(seed)

        ## Order matters: scene() clears everything the randomizer previously
        ## made, lights included, so lights must be created after it.
        if muble_scenes:
            ## Round-robin rather than random, so a corpus of N samples over M
            ## scenes covers every scene evenly instead of leaving some unseen.
            scene_spec = muble_scenes[i % len(muble_scenes)]
            obstacles = place_muble(scene_spec, ds, args.muble_root)
            rnd.scene(obstacles=(0, 0), clear=True)
            centre, radius = muble_bounds(scene_spec)
            rnd.lighting(count=(1, 3), energy=(3.0, 14.0))
            rnd.world()
            frame_camera(camera, rnd, centre,
                         distance=(radius * 2.5, radius * 6.0))
        else:
            obstacles = rnd.scene(obstacles=tuple(args.obstacles), label=OBSTACLE_LABEL)
            rnd.materials(obstacles)
            rnd.lighting(count=(1, 3), energy=(3.0, 14.0))
            rnd.world()
            rnd.pose(bot.root, area=max(1.0, args.extent * 0.3), z=0.15)
            rnd.camera(camera, looking_at=(0, 0, 0.6),
                       distance=(args.extent * 0.4, args.extent * 1.1))
        ## Pin the sampler to the sample's own seed, so a frame does not depend
        ## on how many frames preceded it in this process -- which is what makes
        ## the corpus independent of how it was sharded across workers.
        if hasattr(bpy.context.scene, 'cycles'):
            bpy.context.scene.cycles.seed = seed % (2 ** 31)
        bpy.context.view_layer.update()

        files = {}

        ## 1. photorealistic, on whichever engine can actually light a scene
        set_render_engine(args.rgb_engine)
        if args.rgb_engine == 'cycles':
            configure_cycles(samples=args.cycles_samples, bounces=args.bounces)
        files['rgb'] = quick_render(camera, resolution_x=args.width,
                                    resolution_y=args.height,
                                    output_path=os.path.join(work, '%06d.rgb.png' % i))

        ## 2. geometric passes, which only Cycles can produce
        set_render_engine('cycles')
        configure_cycles(samples=max(1, args.cycles_samples // 4), bounces=0)
        capture = bot.capture(frame=i, resolution=resolution)['front']
        files['depth'] = capture['depth']
        files['segmentation'] = capture['segmentation']

        ## 3. the structural modality
        if not args.no_lineart:
            files['lineart'] = line.render(
                camera, os.path.join(work, '%06d.lineart.png' % i),
                resolution=resolution)

        ds.write(i, seed=seed,
                 meta={'obstacles': len(obstacles),
                       'source': ('muble:%s' % muble_scenes[i % len(muble_scenes)].get('index')
                                  if muble_scenes else 'procedural'),
                       'lens': round(camera.data.lens, 3),
                       ## World space: camera.location is relative to CAM.HUB
                       ## and is not where the camera actually is.
                       'camera': [round(v, 4)
                                  for v in camera.matrix_world.translation],
                       'rgb_engine': args.rgb_engine},
                 **files)

        if (done + 1) % 25 == 0 or done + 1 == len(indices):
            rate = (time.time() - started) / (done + 1)
            print('  %d/%d  %.2f s/sample' % (done + 1, len(indices), rate))

    ds.close()
    elapsed = time.time() - started
    print('wrote %d samples in %.1fs (%.2f s/sample)'
          % (ds.count, elapsed, elapsed / max(1, ds.count)))

    if args.no_verify or sharded:
        ## A shard holds only its own manifest, so it cannot check the corpus.
        ## The launcher verifies after merging.
        return 0

    problems = ds.verify()
    if problems:
        print('VERIFY FAILED: %d problem(s)' % len(problems))
        for p in problems[:10]:
            print('  ' + p)
        if not args.keep_going:
            return 1
    else:
        print('verify: %d samples, all aligned and exposed' % ds.count)
    return 0


#: Obstacles are not part of the robot, so they take an index outside the robot
#: class range rather than colliding with one of its parts.
OBSTACLE_LABEL = 7

#: Objects imported for the current sample, cleared before the next one. Module
#: level because the sample loop is a loop rather than a class, and the previous
#: sample's objects have to be removed by something that remembers them.
_IMPORTED = []


def load_muble(path, root=None):
    """Every MuBlE scene at `path`, which may be a file or a directory."""
    import muble_bridge
    if os.path.isdir(path):
        return [muble_bridge.load(os.path.join(path, name), root=root)
                for name in sorted(os.listdir(path)) if name.endswith('.json')]

    import json
    with open(path) as handle:
        data = json.load(handle)
    if isinstance(data, dict) and 'scenes' in data:
        ## A raw MuBlE bundle holds many scenes; a handoff holds one.
        return [muble_bridge.convert(s, i, root=root)
                for i, s in enumerate(data['scenes'])]
    return [muble_bridge.load(path, root=root)]


def place_muble(scene, ds, root=None):
    """
    Put one MuBlE scene in the world, replacing whatever the last sample left.

    The label map is re-recorded on the Dataset for every sample rather than
    once for the corpus. That is deliberate: scenes contain different objects,
    so the meaning of index 17 genuinely differs between samples, and a single
    corpus-level map would be quietly wrong for most of them.
    """
    import muble_bridge
    for obj in _IMPORTED:
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except (ReferenceError, RuntimeError):
            ## Already gone -- the randomizer's clear() may have taken it.
            pass
    _IMPORTED.clear()

    ## table=False: robotsim's own GROUND already sits with its top face at
    ## z=0, which is the plane MuBlE placed these objects on. Adding the
    ## handoff's table too would put a second surface in the same place and
    ## leave the two z-fighting.
    created, labels = muble_bridge.to_blender(scene, table=False, root=root)
    _IMPORTED.extend(created)
    for token, index in labels.items():
        ds.label(token, index)
    return created


def frame_camera(camera, rnd, target, distance, elevation=(0.35, 1.1),
                 lens=(28.0, 55.0)):
    """
    Aim a camera at a point, in world space.

    `Randomizer.camera` computes its arc in world coordinates and then assigns
    them to `camera.location` and `camera.rotation_euler`, which are *parent*
    space. robotsim's cameras hang off a CAM.HUB with its own offset and a 90
    degree rotation, so the two differ, and the camera ends up somewhere other
    than where the arc put it.

    Over a 10 m procedural scene viewed from 4-11 m that error is small enough
    to be invisible -- every frame still contains the scene. Over a 0.25 m
    tabletop viewed from under a metre it is the whole frame, and the symptom is
    a sample whose segmentation pass contains nothing but ground.

    So this writes `matrix_world` instead, which Blender converts back through
    the parent inverse. The procedural path is deliberately left as it was: its
    framing is already baked into every corpus generated so far, and changing it
    would silently make old and new samples incomparable.
    """
    import mathutils
    target = mathutils.Vector(target)
    azimuth = rnd.rng.uniform(0, 2 * math.pi)
    radius = rnd.rng.uniform(*distance)
    pitch = rnd.rng.uniform(*elevation)
    offset = mathutils.Vector((math.cos(azimuth) * math.cos(pitch),
                               math.sin(azimuth) * math.cos(pitch),
                               math.sin(pitch))) * radius
    position = target + offset
    direction = (target - position).normalized()
    rotation = direction.to_track_quat('-Z', 'Y').to_matrix().to_4x4()
    rotation.translation = position
    camera.matrix_world = rotation
    if hasattr(camera.data, 'lens'):
        camera.data.lens = rnd.rng.uniform(*lens)
    ## The parent inverse is only applied on the next depsgraph evaluation, and
    ## the render reads the evaluated transform.
    bpy.context.view_layer.update()
    return camera


def muble_bounds(scene):
    """
    Centre and radius of a scene's objects, for framing the camera.

    MuBlE tabletops are ~0.1 m across where robotsim's procedural scenes are
    ~10 m. A camera distance tuned for one frames the other as either a dot or
    a texture, so the distance is derived from the content rather than fixed.
    """
    objects = scene.get('objects') or []
    if not objects:
        return (0.0, 0.0, 0.1), 0.5

    points = [o.get('origin', o['position']) for o in objects]
    centre = [sum(p[axis] for p in points) / len(points) for axis in range(3)]
    ## Lift the aim point to mid-object height: aiming at the tabletop puts
    ## every object in the top half of the frame.
    heights = [o['size'][2] for o in objects]
    centre[2] += sum(heights) / len(heights) * 0.5

    radius = 0.0
    for obj, point in zip(objects, points):
        reach = max(obj['size'][0], obj['size'][1]) * 0.5
        span = max(abs(point[axis] - centre[axis]) for axis in range(2))
        radius = max(radius, span + reach)
    return tuple(centre), max(radius, 0.05)

## No __main__ guard: robotsim execs this script into its own globals, so
## __name__ is never '__main__' here. Blender's own argv also carries a '--'
## before the script name, so arguments start after the *last* separator.
_argv = sys.argv[len(sys.argv) - 1 - sys.argv[::-1].index('--') + 1:] if '--' in sys.argv else []
_status = main(_argv)
if _status:
    raise SystemExit(_status)
