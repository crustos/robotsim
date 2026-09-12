#!/usr/bin/env python3
"""
Does the control policy still want more data?

    ./tools/scale_control.py --corpus /tmp/control

Trains the same policy on increasing fractions of the training episodes and
scores every one against the *same* held-out episodes. Holding validation fixed
is the whole point: a curve where both sides change measures nothing, because a
score that moves could be the model getting better or the test getting easier.

This exists because "needs more data" is the easiest thing in the world to
assert and the easiest to be wrong about. On this corpus it was wrong once
already -- doubling the corpus made the policy worse, because the bottleneck was
the architecture destroying the steering signal rather than the corpus being
small. A scaling curve is how that claim gets checked rather than repeated.
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
    files = [f for f in sorted(glob.glob(os.path.join(corpus, 'control*.npz')))
             if not f.endswith('merged.npz')]
    if not files:
        raise SystemExit('no control shards in %s' % corpus)
    return (np.concatenate([np.load(f)['observations'] for f in files]),
            np.concatenate([np.load(f)['actions'] for f in files]),
            np.concatenate([np.load(f)['episodes'] for f in files]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus', default='/tmp/control')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--width', type=int, default=12)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-episodes', type=int, default=18)
    ap.add_argument('--steps', type=int, nargs='*', default=None,
                    help='training-episode counts to try')
    args = ap.parse_args()

    obs, act, eps = load(args.corpus)
    episodes = sorted(set(eps.tolist()))
    held_out = episodes[-args.val_episodes:]
    pool = episodes[:-args.val_episodes]
    val = np.isin(eps, held_out)
    xva, yva = obs[val], act[val]

    sizes = args.steps or [max(1, len(pool) // 4), len(pool) // 2,
                           (3 * len(pool)) // 4, len(pool)]
    print('%d episodes total: %d in the training pool, %d held out (%d frames)\n'
          % (len(episodes), len(pool), len(held_out), len(xva)))

    print('%9s %8s %10s %10s %10s' %
          ('episodes', 'frames', 'train MAE', 'val MAE', 'val omega'))
    baseline = None
    for size in sizes:
        chosen = pool[:size]
        mask = np.isin(eps, chosen)
        xtr, ytr = obs[mask], act[mask]
        if baseline is None:
            ## Fixed to the largest training set's mean, so the reference does
            ## not drift as the training pool grows.
            full = np.isin(eps, pool)
            constant = np.repeat(act[full].mean(axis=0)[None, :], len(yva), axis=0)
            baseline = control_scores(constant, yva)

        net = ControlNet(width=args.width, depth=args.depth,
                         hidden=args.hidden, seed=args.seed)
        train_control(net, xtr, ytr, epochs=args.epochs, batch=8, lr=args.lr,
                      seed=args.seed, log=None)
        train_score = control_scores(net.forward(xtr), ytr)
        val_score = control_scores(net.forward(xva), yva)
        print('%9d %8d %10.3f %10.3f %10.3f'
              % (size, len(xtr), train_score['mae'], val_score['mae'],
                 val_score['omega_mae']))

    print('\nconstant predictor: val MAE %.3f, omega %.3f'
          % (baseline['mae'], baseline['omega_mae']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
