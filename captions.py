"""
Scene descriptions, generated from what the renderer actually drew.

The simulator knows what every object is, where it is and which way up it is.
That knowledge is thrown away at render time: the corpus keeps pixels and an
integer per pixel, and the English word `mug` never appears. This module keeps
it, as a third supervision target beside line art and the semantic map --
propositions about the scene, and the sentences that render them.

    facts = describe(objects, seg, labels)
    facts['text']        # 'A mug is near the camera on the left. A wine
                         #  glass is tipped over on its side.'
    facts['props']       # ['mug:near', 'mug:left', 'wine glass:tipped', ...]

THE RULE THIS MODULE IS BUILT AROUND
------------------------------------
Only assert what is visible in the frame.

The temptation is to describe the scene from the scene graph, because the scene
graph is complete and free. But a caption is supervision for a network whose
only input is the image, and a proposition about an object behind the camera --
or behind a cupboard, or one pixel wide at the far edge -- is a proposition the
network cannot possibly recover. Training on it does not teach the network about
occlusion; it teaches it to guess, because guessing is the only behaviour that
reduces the loss. Every unary fact here is therefore gated on the object's
*rendered* pixel count in the object-index pass, which is the one measurement
that knows about occlusion, clipping, framing and scale at once.

For the same reason image-space facts (left, right, how much of the frame) are
measured from the mask rather than projected from the camera matrix. Projection
gives where an object would be if it were visible; the mask gives where it is.

WHAT A PREDICATE COSTS
----------------------
A predicate with no positive examples cannot be learned and is worse than
useless: it contributes a constant to the loss and a zero to the score. If a
corpus contains no tipped-over objects, `tipped` must not be in the vocabulary.
`vocabulary_from` builds the list from the corpus rather than declaring it up
front, and `coverage` reports how often each proposition actually fires so that
a dead predicate is visible rather than silently diluting the metric.
"""

import math


#: An object must cover at least this fraction of the frame before anything is
#: asserted about it. Set from the smallest thing a person would describe
#: without being asked: below roughly this, a mug is a smudge.
MIN_VISIBLE = 0.004

#: Tilt of an object's own up-axis away from world up, in degrees.
UPRIGHT_MAX = 25.0
TIPPED_MIN = 60.0


def tilt_of(matrix):
    """
    Angle in degrees between an object's local +Z and world +Z.

    Taken from the object's world matrix rather than its euler angles, because
    euler angles are not unique and a parented object's local rotation is not
    its orientation in the world.
    """
    ## Third column of the rotation part is the local +Z axis in world space.
    up = (matrix[0][2], matrix[1][2], matrix[2][2])
    length = math.sqrt(sum(c * c for c in up)) or 1.0
    cosine = max(-1.0, min(1.0, up[2] / length))
    return math.degrees(math.acos(cosine))


def posture(tilt):
    if tilt <= UPRIGHT_MAX:
        return 'upright'
    if tilt >= TIPPED_MIN:
        return 'tipped'
    return 'leaning'


def mask_stats(seg, index):
    """
    Where an object landed in the frame, measured from the index pass.

    Returns (fraction of frame, centre x in [0,1], centre y in [0,1]) or None if
    the object did not survive to the image -- occluded, off-frame, or too small
    to have rasterised at all.
    """
    height, width = seg.shape
    ## Indices are integers written to a float buffer; round before comparing
    ## rather than testing equality on a float.
    hits = (seg.round() == index)
    count = int(hits.sum())
    if not count:
        return None
    ys, xs = hits.nonzero()
    return (count / float(height * width),
            float(xs.mean()) / max(1, width - 1),
            float(ys.mean()) / max(1, height - 1))


def horizontal(cx):
    if cx < 0.38:
        return 'left'
    if cx > 0.62:
        return 'right'
    return 'centre'


def distance_band(depth, near, far):
    """
    Near / mid / far, bucketed against the range present in this scene.

    Relative rather than absolute: a tabletop scene and a room-scale scene have
    nothing in common in metres, and a predicate that means 'within 40 cm' is
    only learnable in a corpus that never changes scale.
    """
    if far - near < 1e-6:
        return 'mid'
    t = (depth - near) / (far - near)
    if t < 0.34:
        return 'near'
    if t > 0.66:
        return 'far'
    return 'mid'


def describe(objects, seg, min_visible=MIN_VISIBLE):
    """
    Propositions and a caption for one rendered frame.

    `objects` is a list of dicts with at least `index`, `label` and `depth`, and
    optionally `tilt`. `seg` is the object-index pass as a 2-D array.

    Returns a dict with `props` (sorted, de-duplicated proposition strings),
    `text`, and `objects` (the per-object records that survived the visibility
    gate, kept so a caller can see what was described and what was dropped).

    Propositions are *existential over category*, not per instance:
    `chopping board:tipped` means some visible chopping board is tipped, and a
    scene with two of them can carry both `chopping board:tipped` and
    `chopping board:upright` without contradiction. That is the right reading
    for a target predicted from a whole frame -- the alternative needs stable
    instance identity across samples, which a randomised scene does not have --
    but it has to be stated, because read as universal the pair looks like a
    bug. The caption disambiguates in words where the proposition cannot.
    """
    visible = []
    for obj in objects:
        stats = mask_stats(seg, obj['index'])
        if stats is None:
            continue
        area, cx, cy = stats
        if area < min_visible:
            continue
        visible.append(dict(obj, area=area, cx=cx, cy=cy))

    if not visible:
        return {'props': [], 'text': 'The scene is empty.', 'objects': []}

    depths = [o['depth'] for o in visible if o.get('depth') is not None]
    near, far = (min(depths), max(depths)) if depths else (0.0, 0.0)

    props = set()
    sentences = []
    ## How many of each category are visible, so the caption can say 'one mug'
    ## and 'the other mug' instead of opening two consecutive sentences with
    ## 'The mug is', which reads as a contradiction rather than as two objects.
    totals = {}
    for obj in visible:
        totals[obj['label']] = totals.get(obj['label'], 0) + 1
    seen = {}

    ## Left to right, so the caption reads in the order a person would scan the
    ## picture rather than in whatever order the scene graph happened to hold.
    for obj in sorted(visible, key=lambda o: o['cx']):
        label = obj['label']
        clauses = []

        side = horizontal(obj['cx'])
        props.add('%s:%s' % (label, side))

        if obj.get('depth') is not None:
            band = distance_band(obj['depth'], near, far)
            props.add('%s:%s' % (label, band))
            clauses.append({'near': 'near the camera',
                            'far': 'far from the camera',
                            'mid': 'at middle distance'}[band])

        if obj.get('tilt') is not None:
            stance = posture(obj['tilt'])
            props.add('%s:%s' % (label, stance))
            if stance == 'tipped':
                clauses.append('tipped over on its side')
            elif stance == 'leaning':
                clauses.append('leaning over')

        where = {'left': 'on the left', 'right': 'on the right',
                 'centre': 'in the middle'}[side]
        clauses.append(where)

        seen[label] = seen.get(label, 0) + 1
        if totals[label] == 1:
            subject = 'The %s' % label
        elif seen[label] == 1:
            subject = 'One %s' % label
        elif seen[label] == totals[label]:
            subject = 'The last %s' % label if totals[label] > 2 else 'The other %s' % label
        else:
            subject = 'Another %s' % label
        sentences.append('%s is %s.' % (subject, ', '.join(clauses)))

    ## A scene-level fact, so the vocabulary is not purely per-object: whether
    ## anything at all has fallen over is the question a tidying policy asks
    ## first, and it is answerable from the whole frame rather than a crop.
    if any(posture(o['tilt']) == 'tipped'
           for o in visible if o.get('tilt') is not None):
        props.add('scene:disordered')
    else:
        props.add('scene:tidy')
        sentences.append('Everything is upright.')

    return {'props': sorted(props), 'text': ' '.join(sentences),
            'objects': [{k: o[k] for k in ('index', 'label', 'area', 'cx', 'cy')}
                        for o in visible]}


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------

def vocabulary_from(entries, min_count=2, max_size=None):
    """
    The proposition vocabulary a corpus can actually support.

    Built from the corpus rather than declared, and filtered by how often each
    proposition fires. A proposition appearing once is not a class, it is a
    coincidence: a network cannot learn it, and including it makes the macro
    score worse for a reason unrelated to the model.
    """
    counts = {}
    for entry in entries:
        for prop in (entry.get('facts') or {}).get('props', []):
            counts[prop] = counts.get(prop, 0) + 1
    keep = [p for p, n in counts.items() if n >= min_count]
    keep.sort(key=lambda p: (-counts[p], p))
    if max_size:
        keep = keep[:max_size]
    return sorted(keep), counts


def coverage(entries, vocabulary):
    """
    How often each proposition holds, as a fraction of samples.

    Reported alongside any score, because a proposition true in 99% of samples
    is predicted correctly by a constant and tells you nothing about the model.
    """
    total = max(1, len(entries))
    counts = {p: 0 for p in vocabulary}
    for entry in entries:
        for prop in (entry.get('facts') or {}).get('props', []):
            if prop in counts:
                counts[prop] += 1
    return {p: counts[p] / total for p in vocabulary}


def encode(props, vocabulary):
    """Proposition list -> multi-hot vector, in vocabulary order."""
    present = set(props)
    return [1.0 if p in present else 0.0 for p in vocabulary]


def decode(vector, vocabulary, threshold=0.5):
    return [p for p, v in zip(vocabulary, vector) if v >= threshold]


def render_text(props):
    """
    A caption from a bare proposition list, for round-tripping.

    Deliberately blunt: the sentences `describe` produces are richer, and this
    exists so that a *predicted* proposition set can be turned back into
    English. That is the direction that matters for a control policy, which will
    consume the network's output rather than the ground truth.
    """
    by_object = {}
    scene = []
    for prop in props:
        subject, _, attribute = prop.partition(':')
        if subject == 'scene':
            scene.append(attribute)
        else:
            by_object.setdefault(subject, []).append(attribute)

    phrase = {'near': 'near the camera', 'far': 'far from the camera',
              'mid': 'at middle distance', 'left': 'on the left',
              'right': 'on the right', 'centre': 'in the middle',
              'tipped': 'tipped over on its side', 'leaning': 'leaning over',
              'upright': 'upright'}
    out = []
    for subject in sorted(by_object):
        clauses = [phrase.get(a, a) for a in sorted(by_object[subject])]
        out.append('The %s is %s.' % (subject, ', '.join(clauses)))
    if 'disordered' in scene:
        out.append('Something has fallen over.')
    elif 'tidy' in scene:
        out.append('Everything is upright.')
    return ' '.join(out) if out else 'Nothing is visible.'


# ---------------------------------------------------------------------------
# Blender
# ---------------------------------------------------------------------------

def objects_from_blender(pairs, camera):
    """
    Per-object records for `describe`, read out of a live Blender scene.

    `pairs` is a sequence of (blender object, index) -- the index being the
    `pass_index` it was given, which is what ties a record to its pixels.

    Imported lazily so the rest of this module stays testable without Blender.
    """
    origin = camera.matrix_world.translation
    out = []
    for obj, index in pairs:
        centre = obj.matrix_world.translation
        depth = math.sqrt(sum((centre[i] - origin[i]) ** 2 for i in range(3)))
        matrix = [[obj.matrix_world[r][c] for c in range(4)] for r in range(4)]
        out.append({'index': int(index),
                    'label': obj.get('label', obj.name),
                    'depth': depth,
                    'tilt': tilt_of(matrix)})
    return out
