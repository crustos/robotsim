#!/usr/bin/env python3
"""
Train the Stage 1 perception network: photograph in, line drawing out.

    ./tools/generate_dataset.py -- --samples 500 --out /tmp/corpus
    ./tools/train_perception.py --corpus /tmp/corpus --epochs 60

Runs under plain Python: `perception.py` needs no `bpy`, so training does not
require Blender and can go somewhere with more cores than the machine that
generated the corpus.

Every score is printed next to the blank-page baseline. On a target that is 99%
white, a network that learns nothing still scores 0.99 accuracy, so accuracy is
shown only to demonstrate that it is useless here; F1 over ink pixels is the
number that means something.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from perception import (Corpus, LineArtNet, MultiModalNet, train,
                        train_multimodal, evaluate, evaluate_multimodal,
                        best_threshold, suggested_ink_weight,
                        suggested_pos_weight, constant_baseline)


def train_with_captions(args, corpus, train_set, val_set, xtr, ytr, xva, yva,
                        weight):
    """
    Train both heads: line art per pixel, propositions per frame.

    The vocabulary is built from the *training* split alone. Building it from
    the whole corpus would let a proposition that only ever appears in
    validation into the target, which is a quiet form of leakage: the model
    would be scored on a class it could not have seen.
    """
    if not train_set.has_facts:
        raise SystemExit('corpus has no captions; regenerate with --tip and '
                         'without --no-captions')

    vocabulary = train_set.vocabulary(min_count=args.min_count)
    if not vocabulary:
        raise SystemExit('no proposition occurs often enough to learn '
                         '(try --min-count 2, or generate more samples)')
    ptr = train_set.propositions(vocabulary)
    pva = val_set.propositions(vocabulary)

    coverage = train_set.coverage(vocabulary)
    print('%d propositions kept (of %d seen); base rates %.2f-%.2f'
          % (len(vocabulary), len(train_set.vocabulary(min_count=1)),
             min(coverage.values()), max(coverage.values())))
    majority = constant_baseline(ptr, pva)
    print('predicting the majority class scores caption micro F1 %.3f, '
          'macro %.3f\n' % (majority['micro_f1'], majority['macro_f1']))

    net = MultiModalNet(n_props=len(vocabulary), width=args.width,
                        depth=args.depth, seed=args.seed)
    if args.resume and os.path.isfile(args.out):
        net.load(args.out)
        print('resumed from %s' % args.out)

    train_multimodal(net, xtr, ytr, ptr, epochs=args.epochs, batch=args.batch,
                     lr=args.lr, ink_weight=weight,
                     pos_weight=suggested_pos_weight(ptr),
                     caption_weight=args.caption_weight, seed=args.seed,
                     val=(xva, yva, pva))

    net.save(args.out)
    with open(os.path.splitext(args.out)[0] + '.vocab.json', 'w') as fh:
        json.dump(vocabulary, fh, indent=1)
    print('\nweights -> %s' % args.out)

    final = evaluate_multimodal(net, xva, yva, pva)
    print('val: line art F1 %.3f (blank %.3f)   caption micro F1 %.3f '
          '(majority %.3f)  macro %.3f  exact-match %.3f'
          % (final['f1'], final['baseline_f1'], final['prop_micro_f1'],
             majority['micro_f1'], final['prop_macro_f1'],
             final['prop_exact_match']))
    if final['prop_micro_f1'] <= majority['micro_f1']:
        print('WARNING: caption head did not beat the majority class.')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--out', default='/tmp/lineart_net.npz')
    ap.add_argument('--width', type=int, default=16, help='channels per layer')
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=6e-3)
    ap.add_argument('--size', type=int, nargs=2, default=(64, 48),
                    help='images are resized to this before training')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--target', default='lineart')
    ap.add_argument('--captions', action='store_true',
                    help='train the caption head as well, on the propositions '
                         'recorded in the corpus manifest')
    ap.add_argument('--caption-weight', type=float, default=1.0)
    ap.add_argument('--min-count', type=int, default=8,
                    help='drop propositions appearing in fewer samples than '
                         'this: a class with almost no positives cannot be '
                         'learned and only drags the macro score down')
    ap.add_argument('--resume', action='store_true',
                    help='continue from --out if it exists')
    ap.add_argument('--tolerance', type=int, default=1,
                    help='matching tolerance in pixels for the tolerant score')
    args = ap.parse_args()

    corpus = Corpus(args.corpus, size=tuple(args.size), limit=args.limit)
    train_set, val_set = corpus.split('train'), corpus.split('val')
    if not len(val_set):
        raise SystemExit('corpus too small to split; generate more samples')
    xtr, ytr = train_set.pairs(target=args.target)
    xva, yva = val_set.pairs(target=args.target)

    ink = float((ytr > 0.5).mean())
    weight = suggested_ink_weight(ytr)
    print('corpus %s: %d train, %d val, %dx%d'
          % (args.corpus, len(xtr), len(xva), args.size[0], args.size[1]))
    print('ink is %.2f%% of pixels -> ink weight %.1f' % (ink * 100, weight))
    print('a blank page scores F1 0.000 and accuracy %.3f on this data\n' % (1 - ink))

    if args.captions:
        return train_with_captions(args, corpus, train_set, val_set,
                                   xtr, ytr, xva, yva, weight)

    net = LineArtNet(width=args.width, depth=args.depth, seed=args.seed)
    if args.resume and os.path.isfile(args.out):
        net.load(args.out)
        print('resumed from %s' % args.out)
    train(net, xtr, ytr, epochs=args.epochs, batch=args.batch, lr=args.lr,
          ink_weight=weight, seed=args.seed, val=(xva, yva))

    ## Chosen on training data, reported on validation: picking it on the
    ## validation set is how a tuned threshold becomes an inflated score.
    threshold, _ = best_threshold(net, xtr, ytr)
    print('\nthreshold %.2f (chosen on train)' % threshold)
    ## Both scores, strict first. Strokes are one pixel wide, so a prediction
    ## that traces a contour perfectly but one pixel off scores zero strictly:
    ## the strict number measures localisation as much as detection, and the
    ## tolerant one -- how boundary detection is normally scored -- separates
    ## them. Neither is quoted without the other.
    for name, (x, y) in (('train', (xtr, ytr)), ('val', (xva, yva))):
        strict = evaluate(net, x, y, threshold)
        loose = evaluate(net, x, y, threshold, tolerance=args.tolerance)
        print('%-5s  strict F1 %.3f (P %.3f R %.3f)   within %dpx F1 %.3f '
              '(P %.3f R %.3f)   blank page %.3f'
              % (name, strict['f1'], strict['precision'], strict['recall'],
                 args.tolerance, loose['f1'], loose['precision'], loose['recall'],
                 strict['baseline_f1']))

    net.save(args.out)
    print('\nweights -> %s' % args.out)
    final = evaluate(net, xva, yva, threshold)
    if final['f1'] <= final['baseline_f1']:
        print('WARNING: did not beat the blank page. Nothing was learned.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
