#!/usr/bin/env python3
"""
Stage 1 tests. Runs under plain Python -- perception.py needs no bpy, and a
corpus is faked here with PIL so the test does not need Blender either.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from perception import (Corpus, LineArtNet, Conv2d, im2col, col2im, weighted_bce,
                        sigmoid, scores, blank_baseline, suggested_ink_weight,
                        train, evaluate, best_threshold)

print('hello perception test...')
OUT = '/tmp/robotsim-perception-test'


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_gradients():
    """
    The hand-written backward pass must agree with a numerical derivative.

    Run in float64 deliberately. A numerical gradient is a difference of two
    nearly equal numbers, and in float32 the cancellation swamps the result --
    the check then fails on arithmetic rather than on a wrong gradient, which is
    a genuinely confusing hour to spend.
    """
    rng = np.random.default_rng(0)
    net = LineArtNet(width=5, depth=3, seed=1, dtype=np.float64)
    x = rng.random((2, 3, 7, 6))
    y = (rng.random((2, 1, 7, 6)) > 0.85).astype(np.float64)

    _loss, grad = weighted_bce(net.forward(x), y, 5.0)
    net.backward(grad)
    analytic = [g.copy() for _p, g in net.params()]

    eps, worst = 1e-6, 0.0
    for pi, (p, _g) in enumerate(net.params()):
        flat = p.reshape(-1)
        for k in rng.choice(flat.size, size=min(12, flat.size), replace=False):
            original = flat[k]
            flat[k] = original + eps
            plus = weighted_bce(net.forward(x), y, 5.0)[0]
            flat[k] = original - eps
            minus = weighted_bce(net.forward(x), y, 5.0)[0]
            flat[k] = original
            numeric = (plus - minus) / (2 * eps)
            exact = analytic[pi].reshape(-1)[k]
            worst = max(worst, abs(numeric - exact) / max(1e-12, abs(numeric) + abs(exact)))
    print('worst relative gradient error: %.2e' % worst)
    assert worst < 1e-4, worst
    print('gradients OK')


def test_im2col_adjoint():
    """col2im must be the exact adjoint of im2col, or the conv gradient is wrong."""
    rng = np.random.default_rng(1)
    a = rng.random((2, 3, 5, 4))
    b = rng.random((2, 3 * 9, 20))
    lhs = float((im2col(a, 3, 1) * b).sum())
    rhs = float((a * col2im(b, a.shape, 3, 1)).sum())
    print('adjoint: %.6f vs %.6f' % (lhs, rhs))
    assert abs(lhs - rhs) / max(1.0, abs(lhs)) < 1e-9
    print('adjoint OK')


def test_blank_page_is_the_baseline():
    """
    The trap this whole module is built around.

    A blank prediction on sparse line art scores near-perfect accuracy and zero
    F1. Any result that does not beat this has learned that paper is white.
    """
    y = np.zeros((4, 1, 32, 32), dtype=np.float32)
    y[:, :, 10, :] = 1.0                      # about 3% ink
    base = blank_baseline(y)
    print('blank page: accuracy %.3f, F1 %.3f' % (base['accuracy'], base['f1']))
    assert base['accuracy'] > 0.95, base
    assert base['f1'] == 0.0, base
    ## a perfect prediction scores 1
    assert close(scores(y, y)['f1'], 1.0)
    ## and the suggested weight rebalances toward ink, but is capped
    w = suggested_ink_weight(y)
    assert 1.0 < w <= 40.0, w
    assert suggested_ink_weight(np.zeros_like(y)) == 1.0, 'no ink at all must not divide by zero'
    print('baseline OK')


def test_loss_is_stable():
    """BCE from logits must not overflow where a confident prediction lives."""
    logits = np.array([[[[-800.0, 800.0], [0.0, -40.0]]]], dtype=np.float64)
    targets = np.array([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=np.float64)
    loss, grad = weighted_bce(logits, targets, 10.0)
    print('extreme logits -> loss %.4f' % loss)
    assert np.isfinite(loss), loss
    assert np.all(np.isfinite(grad)), grad
    assert np.all(np.isfinite(sigmoid(logits)))
    assert close(float(sigmoid(np.array([-800.0]))[0]), 0.0)
    print('loss stability OK')


def fake_corpus(root, samples=16, size=(24, 18)):
    """
    A tiny learnable corpus: a bright rectangle on a dark field, and its outline.

    Synthetic rather than generated in Blender so this test needs neither a
    render nor a GPU -- and the mapping is genuinely learnable from local
    evidence, which is what the network is being asked to do for real.
    """
    from PIL import Image, ImageDraw
    os.makedirs(os.path.join(root, 'train'), exist_ok=True)
    rng = np.random.default_rng(7)
    entries = []
    for i in range(samples):
        w, h = size
        rgb = Image.new('RGB', size, (20, 20, 30))
        line = Image.new('L', size, 255)
        dr, dl = ImageDraw.Draw(rgb), ImageDraw.Draw(line)
        x0 = int(rng.integers(2, w // 2)); y0 = int(rng.integers(2, h // 2))
        x1 = int(rng.integers(x0 + 4, w - 1)); y1 = int(rng.integers(y0 + 4, h - 1))
        dr.rectangle([x0, y0, x1, y1], fill=(200, 190, 180))
        dl.rectangle([x0, y0, x1, y1], outline=0)
        rgb_path = os.path.join(root, 'train', '%06d.rgb.png' % i)
        line_path = os.path.join(root, 'train', '%06d.lineart.png' % i)
        rgb.save(rgb_path); line.save(line_path)
        entries.append({'index': i, 'split': 'train', 'labels': {},
                        'files': {'rgb': 'train/%06d.rgb.png' % i,
                                  'lineart': 'train/%06d.lineart.png' % i}})
    with open(os.path.join(root, 'manifest.jsonl'), 'w') as fh:
        for e in entries:
            fh.write(json.dumps(e) + '\n')
    return root


def test_corpus_and_split():
    """The manifest defines the samples, and the split is deterministic."""
    os.system('rm -rf %s' % OUT)
    fake_corpus(OUT, samples=20)
    c = Corpus(OUT)
    assert len(c) == 20, len(c)
    tr, va = c.split('train'), c.split('val')
    assert len(tr) == 16 and len(va) == 4, (len(tr), len(va))
    ## by index, so it is identical across runs and processes
    assert [e['index'] for e in va.entries] == [16, 17, 18, 19]
    assert Corpus(OUT).split('val').entries == va.entries

    x, y = tr.pairs()
    assert x.shape[1] == 3 and y.shape[1] == 1, (x.shape, y.shape)
    assert x.shape[0] == 16
    assert 0.0 <= x.min() and x.max() <= 1.0
    ## targets are inverted so 1 means ink
    assert y.mean() < 0.5, 'most of a line drawing should be background'
    print('corpus OK: %s -> x %s y %s' % (c, x.shape, y.shape))


def test_it_actually_learns():
    """
    End to end on a learnable task: the network must beat the blank page.

    This is the assertion that matters. Loss going down proves nothing here --
    the fastest way down is to predict background everywhere.
    """
    fake_corpus(OUT, samples=24)
    c = Corpus(OUT)
    tr, va = c.split('train'), c.split('val')
    xtr, ytr = tr.pairs()
    xva, yva = va.pairs()

    net = LineArtNet(width=8, depth=3, seed=0)
    before = evaluate(net, xva, yva)
    history = train(net, xtr, ytr, epochs=25, batch=4, lr=1e-2, log=None)
    after = evaluate(net, xva, yva)

    print('val F1 %.3f -> %.3f   (blank page %.3f, accuracy %.3f)'
          % (before['f1'], after['f1'], after['baseline_f1'], after['accuracy']))
    assert history[-1]['loss'] < history[0]['loss'], 'loss did not fall'
    assert after['f1'] > after['baseline_f1'], 'did not beat the blank page'
    assert after['f1'] > 0.3, 'learned too little: F1 %.3f' % after['f1']
    assert after['recall'] > 0.2, after

    ## threshold selection returns something sane and does not crash
    t, f1 = best_threshold(net, xtr, ytr)
    assert 0.0 < t < 1.0, t
    print('learning OK')


def test_save_load_round_trip():
    """Weights must survive a round trip, or a trained model is unusable."""
    fake_corpus(OUT, samples=8)
    c = Corpus(OUT)
    x, y = c.pairs()
    net = LineArtNet(width=6, depth=3, seed=3)
    train(net, x, y, epochs=3, batch=4, log=None)
    before = net.predict(x)
    path = net.save(os.path.join(OUT, 'net.npz'))

    restored = LineArtNet(width=6, depth=3, seed=99)   # different init
    assert not np.allclose(restored.predict(x), before)
    restored.load(path)
    assert np.allclose(restored.predict(x), before, atol=1e-6)
    print('save/load OK')


test_gradients()
test_im2col_adjoint()
test_blank_page_is_the_baseline()
test_loss_is_stable()
test_corpus_and_split()
test_it_actually_learns()
test_save_load_round_trip()
print('perception test OK')
