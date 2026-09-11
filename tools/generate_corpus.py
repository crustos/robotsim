#!/usr/bin/env python3
"""
Generate a corpus across several Blender processes.

    ./tools/generate_corpus.py --samples 5000 --out /data/corpus --workers 8

Generation is embarrassingly parallel: each sample is an independent scene built
from its own seed, so N workers produce the corpus in close to 1/N the time. The
work is CPU-bound inside Blender, and Blender does not release the GIL in a way
that threads would help with, so the workers are separate processes.

WHAT THE WORKERS SHARE
----------------------
Nothing, by construction. Worker `k` generates the samples where
`index % workers == k`, derives every random choice from a seed computed from
that index, and writes its own manifest. Two consequences worth stating:

  * the scene content of a sample depends on its index and not on which
    process took it, so the geometry, materials, lighting and viewpoint are
    identical whatever the worker count.

    Measured, that holds exactly for the photorealistic, depth and segmentation
    passes -- bit-identical across worker counts. The line pass does not: it is
    reproducible run-to-run at a *fixed* worker count, but Blender's stroke
    renderer carries state between renders within a process, so changing how
    the samples are divided shifts a few strokes by up to one pixel (mean
    0.5/255 over the image). The worker count is therefore recorded in
    `dataset.json`: reproducing a corpus exactly means reproducing the seed
    *and* the shard layout.
  * nothing appends to a shared file, so there is no interleaving to get
    subtly wrong under load.

The shard manifests are merged and sorted by index afterwards, then the whole
corpus is verified once -- a shard cannot verify what it cannot see.

A NOTE ON EXPECTED SPEEDUP
--------------------------
Close to linear in cores, until memory or disk bandwidth binds. Each Blender
process holds its own copy of the scene, so peak memory is roughly `workers`
times a single process; on a machine with few gigabytes, more workers than cores
will swap rather than scale.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='/tmp/corpus')
    ap.add_argument('--samples', type=int, default=256)
    ap.add_argument('--workers', type=int, default=0,
                    help='0 means one per CPU')
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--height', type=int, default=96)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cycles-samples', type=int, default=16)
    ap.add_argument('--rgb-engine', default='eevee')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--prune', action='store_true',
                    help='drop samples that fail verification instead of failing')
    args = ap.parse_args()

    workers = args.workers or (os.cpu_count() or 1)
    workers = max(1, min(workers, args.samples))

    out = os.path.abspath(args.out)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, 'train'), exist_ok=True)

    generator = os.path.join(HERE, 'generate_dataset.py')
    headless = os.path.join(ROOT, 'headless.py')
    common = ['--out', out, '--samples', str(args.samples),
              '--width', str(args.width), '--height', str(args.height),
              '--seed', str(args.seed),
              '--cycles-samples', str(args.cycles_samples),
              '--rgb-engine', args.rgb_engine]

    print('generating %d samples with %d worker(s) -> %s'
          % (args.samples, workers, out))
    started = time.time()
    processes = []
    logs = []
    for shard in range(workers):
        ## headless.py is an sh/python polyglot that re-execs itself through
        ## Blender. Run it through sh explicitly rather than relying on its
        ## execute bit, which this repo leaves to `make install` -- the launcher
        ## should work on a fresh clone.
        cmd = ['/bin/sh', headless, generator, '--'] + common + [
            '--shard', str(shard), '--shards', str(workers)]
        log = open(os.path.join('/tmp', 'corpus-shard-%03d.log' % shard), 'w')
        logs.append(log)
        processes.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))

    failed = []
    for shard, proc in enumerate(processes):
        code = proc.wait()
        logs[shard].close()
        if code != 0:
            failed.append((shard, code))
    elapsed = time.time() - started

    if failed:
        print('FAILED shards: %s' % failed)
        for shard, _code in failed[:3]:
            path = '/tmp/corpus-shard-%03d.log' % shard
            print('--- tail of %s ---' % path)
            with open(path) as fh:
                print(''.join(fh.readlines()[-12:]))
        return 1

    from dataset import Dataset, merge_manifests
    ## With one worker the generator has already written the real manifest;
    ## merging is a no-op that must not touch it.
    merged = merge_manifests(out) if workers > 1 else len(
        Dataset(out, overwrite=False).entries())
    print('merged %d manifest entries in %.1fs (%.2f s/sample wall)'
          % (merged, elapsed, elapsed / max(1, args.samples)))

    ds = Dataset(out, passes=('rgb', 'depth', 'segmentation', 'lineart'),
                 overwrite=False)
    problems = ds.verify()
    if problems and args.prune:
        from dataset import indices_in
        dropped = ds.prune(indices_in(problems))
        print('pruned %d unusable sample(s) of %d:' % (dropped, args.samples))
        for p in problems[:5]:
            print('  ' + p)
        problems = ds.verify()
    if problems:
        print('VERIFY FAILED: %d problem(s)' % len(problems))
        for p in problems[:10]:
            print('  ' + p)
        return 1
    ## Record how the corpus was produced. The seed alone does not reproduce
    ## it -- see the note on the line pass above -- so the worker count belongs
    ## in the corpus, not in whoever remembers the command they ran.
    info_path = os.path.join(out, Dataset.INFO)
    try:
        import json
        with open(info_path) as fh:
            info = json.load(fh)
    except (OSError, ValueError):
        info = {}
    info['generation'] = {
        'samples': args.samples, 'workers': workers, 'seed': args.seed,
        'width': args.width, 'height': args.height,
        'rgb_engine': args.rgb_engine, 'cycles_samples': args.cycles_samples,
        'wall_seconds': round(elapsed, 1),
    }
    info['samples'] = len(ds.entries())
    with open(info_path, 'w') as fh:
        json.dump(info, fh, indent=2, sort_keys=True)

    print('verify: %d samples, all aligned and exposed' % len(ds.entries()))
    return 0


sys.exit(main())
