#!/usr/bin/env python3
"""
Train the Stage 2 control policy: abstraction in, drive command out.

    ./tools/generate_control.py -- --episodes 30 --out /tmp/control
    ./tools/train_control.py --corpus /tmp/control

Runs under plain Python. Stage 2 never sees a photograph, so training needs
neither Blender nor the photorealistic pass -- which is the practical half of
the architecture's argument: the expensive renderer is needed once, to make the
corpus, and never again.

THE SPLIT IS BY EPISODE
-----------------------
Consecutive frames in a rollout are nearly the same picture with nearly the same
command. Splitting those at random puts a frame's near-duplicate on the other
side of the wall, and the validation score then measures memorisation. Splitting
by episode is the only split that means anything here, and it is why the
reported numbers are lower than a frame-split would give.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from control import ControlNet, control_scores, train_control


def load(corpus):
    files = sorted(glob.glob(os.path.join(corpus, 'control*.npz')))
    files = [f for f in files if not f.endswith('merged.npz')]
    if not files:
        raise SystemExit('no control shards in %s' % corpus)
    obs = np.concatenate([np.load(f)['observations'] for f in files])
    act = np.concatenate([np.load(f)['actions'] for f in files])
    eps = np.concatenate([np.load(f)['episodes'] for f in files])
    return obs, act, eps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus', default='/tmp/control')
    ap.add_argument('--out', default='/tmp/control_net.npz')
    ap.add_argument('--width', type=int, default=12)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fraction', type=float, default=0.75)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    obs, act, eps = load(args.corpus)
    episodes = sorted(set(eps.tolist()))
    cut = max(1, int(len(episodes) * args.fraction))
    train_mask = np.isin(eps, episodes[:cut])
    val_mask = np.isin(eps, episodes[cut:])
    if not val_mask.any():
        raise SystemExit('not enough episodes to hold any out')

    xtr, ytr = obs[train_mask], act[train_mask]
    xva, yva = obs[val_mask], act[val_mask]
    print('corpus %s: %d frames over %d episodes'
          % (args.corpus, len(obs), len(episodes)))
    print('train %d frames / %d episodes, val %d frames / %d episodes'
          % (len(xtr), cut, len(xva), len(episodes) - cut))

    ## The number to beat. Predicting the training mean is the control-task
    ## equivalent of the blank page: an expert that mostly drives forward makes
    ## the mean command a decent guess, and a policy that has learned nothing
    ## still posts a small error.
    constant = np.repeat(ytr.mean(axis=0)[None, :], len(yva), axis=0)
    baseline = control_scores(constant, yva)
    print('predicting the training mean gives val MAE %.3f '
          '(v %.3f, omega %.3f)\n'
          % (baseline['mae'], baseline['v_mae'], baseline['omega_mae']))

    net = ControlNet(width=args.width, depth=args.depth, hidden=args.hidden,
                     seed=args.seed)
    if args.resume and os.path.isfile(args.out):
        net.load(args.out)
        print('resumed from %s' % args.out)

    train_control(net, xtr, ytr, epochs=args.epochs, batch=args.batch,
                  lr=args.lr, seed=args.seed, val=(xva, yva))

    net.save(args.out)
    print('\nweights -> %s' % args.out)

    final = control_scores(net.forward(xva), yva)
    improvement = 1.0 - final['mae'] / max(1e-9, baseline['mae'])
    print('val MAE %.3f vs %.3f for the training mean  (%.0f%% of the '
          'constant predictor\'s error removed)'
          % (final['mae'], baseline['mae'], improvement * 100))
    print('  v     MAE %.3f (mean %.3f)' % (final['v_mae'], baseline['v_mae']))
    print('  omega MAE %.3f (mean %.3f)' % (final['omega_mae'],
                                            baseline['omega_mae']))
    if final['mae'] >= baseline['mae']:
        print('WARNING: did not beat the constant predictor. Nothing learned.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
