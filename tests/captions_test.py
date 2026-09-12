#!/usr/bin/env python3
"""
Tests for scene descriptions and the multi-modal perception head.

Runs under plain python3. `captions.py` is free of `bpy` except for one lazy
helper, and `perception.py` is free of it entirely, so both are testable without
launching Blender.

    ./captions_test.py          # or: make test_captions
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import captions as C                                          # noqa: E402
import perception as P                                        # noqa: E402

print('hello captions test...')


def mask(shape, boxes):
    """A synthetic object-index pass: {index: (y0, y1, x0, x1)}."""
    seg = np.zeros(shape, dtype=np.float32)
    for index, (y0, y1, x0, x1) in boxes.items():
        seg[y0:y1, x0:x1] = index
    return seg


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# visibility
# ---------------------------------------------------------------------------

def test_invisible_objects_are_not_described():
    """
    The rule the module exists to enforce.

    An object with no pixels is not mentioned, however complete the scene graph
    is. Training on a proposition the image cannot support does not teach a
    network about occlusion -- it teaches it to guess, because guessing is the
    only behaviour that lowers the loss.
    """
    seg = mask((48, 64), {17: (10, 30, 5, 25)})
    objects = [{'index': 17, 'label': 'mug', 'depth': 1.0, 'tilt': 0.0},
               {'index': 18, 'label': 'ghost', 'depth': 1.0, 'tilt': 90.0}]
    facts = C.describe(objects, seg)
    assert 'mug' in facts['text'], facts['text']
    assert 'ghost' not in facts['text'], facts['text']
    assert not any(p.startswith('ghost') for p in facts['props']), facts['props']
    ## And the scene-level fact follows the visible set, not the scene graph:
    ## the only tipped object is invisible, so the scene is tidy.
    assert 'scene:tidy' in facts['props'], facts['props']
    print('  invisible object dropped, scene-level fact follows the visible set')


def test_tiny_objects_are_gated_out():
    """A few pixels is not a description; it is noise with a label."""
    seg = mask((48, 64), {17: (10, 30, 5, 25), 19: (0, 2, 0, 2)})
    objects = [{'index': 17, 'label': 'mug', 'depth': 1.0, 'tilt': 0.0},
               {'index': 19, 'label': 'speck', 'depth': 1.0, 'tilt': 0.0}]
    facts = C.describe(objects, seg)
    assert len(facts['objects']) == 1, facts['objects']
    assert facts['objects'][0]['label'] == 'mug'
    ## Lower the gate and it comes back -- the threshold is doing the work,
    ## not an accident of the mask.
    permissive = C.describe(objects, seg, min_visible=0.0)
    assert len(permissive['objects']) == 2, permissive['objects']
    print('  4-pixel object gated out, and returns when the gate is lowered')


def test_empty_scene():
    facts = C.describe([], np.zeros((8, 8), dtype=np.float32))
    assert facts['props'] == [], facts
    assert 'empty' in facts['text'].lower(), facts['text']
    print('  empty scene: %r' % facts['text'])


# ---------------------------------------------------------------------------
# geometry -> words
# ---------------------------------------------------------------------------

def test_posture_from_tilt():
    assert C.posture(0.0) == 'upright'
    assert C.posture(90.0) == 'tipped'
    assert C.posture(45.0) == 'leaning'
    ## Tilt comes from the world matrix's third column, not from euler angles:
    ## a matrix whose local +Z lies along world +X is fully tipped.
    on_its_side = [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]
    assert close(C.tilt_of(on_its_side), 90.0, 1e-4), C.tilt_of(on_its_side)
    upright = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert close(C.tilt_of(upright), 0.0, 1e-4)
    print('  tilt read from the world matrix, not euler angles')


def test_side_comes_from_the_mask():
    """
    Image-space facts are measured, not projected.

    Projection says where an object would be if it were visible; the mask says
    where it is. The two differ exactly when it matters.
    """
    seg = mask((48, 64), {17: (10, 30, 0, 12), 18: (10, 30, 52, 64)})
    objects = [{'index': 17, 'label': 'mug', 'depth': 1.0},
               {'index': 18, 'label': 'bowl', 'depth': 1.0}]
    props = C.describe(objects, seg)['props']
    assert 'mug:left' in props, props
    assert 'bowl:right' in props, props
    print('  left/right taken from mask centroids')


def test_distance_is_relative_to_the_scene():
    """
    Near and far are bucketed against the range present, not against metres.

    A tabletop and a room have nothing in common in absolute units, and a
    predicate meaning 'within 40cm' is only learnable in a corpus that never
    changes scale.
    """
    seg = mask((48, 64), {17: (10, 30, 2, 18), 18: (10, 30, 24, 40),
                          19: (10, 30, 46, 62)})
    def props_for(depths):
        objects = [{'index': 17 + k, 'label': 'obj%d' % k, 'depth': d}
                   for k, d in enumerate(depths)]
        return C.describe(objects, seg)['props']

    tabletop = props_for([0.30, 0.50, 0.70])
    room = props_for([3.0, 5.0, 7.0])
    for a, b in zip(sorted(p for p in tabletop if ':' in p),
                    sorted(p for p in room if ':' in p)):
        assert a == b, (a, b)
    print('  same relative layout gives the same words at 10x the scale')


def test_duplicate_categories_read_as_separate_objects():
    """
    Propositions are existential over category; the caption says so in words.

    Two chopping boards, one tipped, legitimately produce both
    `chopping board:tipped` and `chopping board:upright`. Read as universal that
    looks like a contradiction, so the sentences must not open with 'The
    chopping board is' twice.
    """
    seg = mask((48, 64), {17: (10, 30, 2, 18), 18: (10, 30, 24, 40)})
    objects = [{'index': 17, 'label': 'board', 'depth': 1.0, 'tilt': 2.0},
               {'index': 18, 'label': 'board', 'depth': 1.2, 'tilt': 89.0}]
    facts = C.describe(objects, seg)
    assert 'board:tipped' in facts['props'], facts['props']
    assert 'board:upright' in facts['props'], facts['props']
    assert facts['text'].count('The board is') == 0, facts['text']
    assert 'One board' in facts['text'], facts['text']
    print('  %s' % facts['text'])


def test_round_trip_through_text():
    """A predicted proposition set can be turned back into English."""
    props = ['mug:near', 'mug:left', 'scene:disordered', 'bowl:tipped']
    text = C.render_text(props)
    assert 'mug' in text and 'bowl' in text, text
    assert 'fallen' in text, text
    print('  %s' % text)


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------

def test_rare_propositions_are_dropped():
    """
    A proposition seen once is a coincidence, not a class.

    It cannot be learned, it contributes a constant to the loss, and leaving it
    in the vocabulary drags the macro score down for a reason unrelated to the
    model.
    """
    entries = [{'facts': {'props': ['mug:near', 'scene:tidy']}} for _ in range(5)]
    entries.append({'facts': {'props': ['unicorn:tipped']}})
    vocab, counts = C.vocabulary_from(entries, min_count=2)
    assert 'mug:near' in vocab, vocab
    assert 'unicorn:tipped' not in vocab, vocab
    assert counts['unicorn:tipped'] == 1
    print('  vocabulary %s (dropped a singleton)' % vocab)


def test_coverage_and_encoding():
    entries = [{'facts': {'props': ['a:1', 'b:2']}},
               {'facts': {'props': ['a:1']}}]
    vocab = ['a:1', 'b:2']
    cov = C.coverage(entries, vocab)
    assert close(cov['a:1'], 1.0) and close(cov['b:2'], 0.5), cov
    assert C.encode(['b:2'], vocab) == [0.0, 1.0]
    assert C.decode([0.9, 0.1], vocab) == ['a:1']
    print('  coverage %s' % cov)


# ---------------------------------------------------------------------------
# the multi-task head
# ---------------------------------------------------------------------------

def test_multimodal_gradients():
    """
    Analytic gradients agree with a numerical check, in double precision.

    Double deliberately: a numerical derivative is a difference of two nearly
    equal numbers, and in single precision the cancellation swamps the result,
    so the check fails on arithmetic rather than on a wrong gradient.
    """
    rng = np.random.default_rng(0)
    net = P.MultiModalNet(n_props=5, width=4, depth=3, seed=1, dtype=np.float64)
    x = rng.standard_normal((2, 3, 7, 9))
    y_ink = (rng.random((2, 1, 7, 9)) > 0.8).astype(np.float64)
    y_prop = (rng.random((2, 5)) > 0.5).astype(np.float64)

    def total():
        ink, prop = net.forward(x)
        return (P.weighted_bce(ink, y_ink, 5.0)[0]
                + P.proposition_bce(prop, y_prop)[0])

    ink, prop = net.forward(x)
    net.backward(P.weighted_bce(ink, y_ink, 5.0)[1],
                 P.proposition_bce(prop, y_prop)[1])
    analytic = [g.copy() for _p, g in net.params()]

    eps, worst = 1e-6, 0.0
    for k, (param, _g) in enumerate(net.params()):
        flat, grad = param.reshape(-1), analytic[k].reshape(-1)
        for i in range(0, flat.size, max(1, flat.size // 5)):
            old = flat[i]
            flat[i] = old + eps
            up = total()
            flat[i] = old - eps
            down = total()
            flat[i] = old
            numeric = (up - down) / (2 * eps)
            worst = max(worst, abs(numeric - grad[i])
                        / max(1e-9, abs(numeric) + abs(grad[i])))
    assert worst < 1e-6, worst
    print('  worst relative gradient error %.2e' % worst)


def test_both_heads_share_one_backward_pass():
    """
    The trunk sees the sum of both heads' gradients, not two passes.

    Conv2d assigns to `dw` rather than accumulating, so backpropagating through
    the trunk twice would overwrite the first head's parameter gradients before
    the optimiser ever saw them -- a bug that trains fine and silently ignores
    one of the two tasks.
    """
    net = P.MultiModalNet(n_props=3, width=4, depth=3, seed=2, dtype=np.float64)
    x = np.random.default_rng(1).standard_normal((2, 3, 6, 6))
    ink, prop = net.forward(x)

    d_ink = np.ones_like(ink) * 0.01
    d_prop = np.ones_like(prop) * 0.01
    net.backward(d_ink, None)
    ink_only = net.trunk[0].dw.copy()
    net.forward(x)
    net.backward(np.zeros_like(ink), d_prop)
    prop_only = net.trunk[0].dw.copy()
    net.forward(x)
    net.backward(d_ink, d_prop)
    both = net.trunk[0].dw.copy()

    assert np.allclose(both, ink_only + prop_only, atol=1e-10), \
        np.abs(both - ink_only - prop_only).max()
    ## And the caption head genuinely reaches the trunk.
    assert np.abs(prop_only).max() > 1e-12, 'caption head does not reach the trunk'
    print('  trunk gradient is the sum of both heads')


def test_proposition_metrics_expose_the_constant_predictor():
    """
    The blank page has an exact analogue here and must be reported beside it.

    Most propositions are false in most frames, so answering 'no' to everything
    scores high accuracy. `constant_baseline` is what makes that visible.
    """
    truth = np.zeros((10, 3), dtype=np.float32)
    truth[:, 0] = 1.0                      # always true
    truth[:3, 1] = 1.0                     # sometimes
    always_no = np.zeros_like(truth)
    scored = P.proposition_scores(always_no, truth)
    assert close(scored['micro_f1'], 0.0), scored
    base = P.constant_baseline(truth, truth)
    ## Predicting the majority class gets the always-true proposition for free.
    assert base['micro_f1'] > 0.5, base
    perfect = P.proposition_scores(truth, truth)
    assert close(perfect['micro_f1'], 1.0), perfect
    assert close(perfect['exact_match'], 1.0), perfect
    print('  all-negative F1 %.3f, majority %.3f, perfect %.3f'
          % (scored['micro_f1'], base['micro_f1'], perfect['micro_f1']))


def test_pos_weight_is_capped():
    """
    A proposition with almost no positives must not dominate every gradient.

    Its inverse base rate would otherwise be in the hundreds, for a class the
    model has almost no evidence about.
    """
    targets = np.zeros((500, 2), dtype=np.float32)
    targets[:250, 0] = 1.0                 # balanced
    targets[0, 1] = 1.0                    # one positive in five hundred
    weight = P.suggested_pos_weight(targets, cap=20.0)
    assert close(float(weight[0]), 1.0, 1e-3), weight
    assert close(float(weight[1]), 20.0, 1e-3), weight
    print('  pos weights %s (capped)' % weight.round(2).tolist())


TESTS = [test_invisible_objects_are_not_described, test_tiny_objects_are_gated_out,
         test_empty_scene, test_posture_from_tilt, test_side_comes_from_the_mask,
         test_distance_is_relative_to_the_scene,
         test_duplicate_categories_read_as_separate_objects,
         test_round_trip_through_text, test_rare_propositions_are_dropped,
         test_coverage_and_encoding, test_multimodal_gradients,
         test_both_heads_share_one_backward_pass,
         test_proposition_metrics_expose_the_constant_predictor,
         test_pos_weight_is_capped]


if __name__ == '__main__':
    failed = 0
    for test in TESTS:
        print('%s:' % test.__name__)
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print('  FAILED: %s' % exc)
    print('\n%d/%d passed' % (len(TESTS) - failed, len(TESTS)))
    sys.exit(1 if failed else 0)
