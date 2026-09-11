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

    ds = Dataset(args.out, passes=tuple(passes), split=args.split,
                 overwrite=True, note='robotsim procedural corpus')
    for token, index in SEGMENT_CLASSES.items():
        ds.label(token, index)
    ds.label('obstacle', OBSTACLE_LABEL)

    print('generating %d samples -> %s' % (args.samples, args.out))
    started = time.time()
    for i in range(args.samples):
        seed = args.seed * 1000003 + i
        rnd.reseed(seed)

        ## Order matters: scene() clears everything the randomizer previously
        ## made, lights included, so lights must be created after it.
        obstacles = rnd.scene(obstacles=tuple(args.obstacles), label=OBSTACLE_LABEL)
        rnd.materials(obstacles)
        rnd.lighting(count=(1, 3), energy=(3.0, 14.0))
        rnd.world()
        rnd.pose(bot.root, area=max(1.0, args.extent * 0.3), z=0.15)
        rnd.camera(camera, looking_at=(0, 0, 0.6),
                   distance=(args.extent * 0.4, args.extent * 1.1))
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
                       'lens': round(camera.data.lens, 3),
                       'camera': [round(v, 4) for v in camera.location],
                       'rgb_engine': args.rgb_engine},
                 **files)

        if (i + 1) % 25 == 0 or i + 1 == args.samples:
            rate = (time.time() - started) / (i + 1)
            print('  %d/%d  %.2f s/sample' % (i + 1, args.samples, rate))

    ds.close()
    elapsed = time.time() - started
    print('wrote %d samples in %.1fs (%.2f s/sample)'
          % (ds.count, elapsed, elapsed / max(1, ds.count)))

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

## No __main__ guard: robotsim execs this script into its own globals, so
## __name__ is never '__main__' here. Blender's own argv also carries a '--'
## before the script name, so arguments start after the *last* separator.
_argv = sys.argv[len(sys.argv) - 1 - sys.argv[::-1].index('--') + 1:] if '--' in sys.argv else []
_status = main(_argv)
if _status:
    raise SystemExit(_status)
