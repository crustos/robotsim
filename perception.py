"""
Stage 1: learning to turn a photograph into a line drawing.

This is the perception half of the chained architecture. It takes the messy
photorealistic frame and emits the structural abstraction a control policy is
trained on -- so that the policy never sees a texture, a shadow or a specular
highlight, and cannot be confused by one.

    corpus = Corpus('/tmp/corpus')
    net = LineArtNet(width=12)
    train(net, corpus, epochs=20)
    print(evaluate(net, corpus.split('val')))

WHY NUMPY AND NOT TORCH
-----------------------
Torch is the right tool and this module is deliberately shaped so it can be
swapped in: `LineArtNet` is a plain stack of layers behind `forward`/`backward`,
and nothing above it assumes how the gradients were produced.

It is not used *here* because the PyPI Linux wheel links CUDA libraries even for
CPU-only use -- `libtorch_global_deps.so` needs libcublas at import -- and the
dependency chain measures over four gigabytes unpacked, which does not fit this
container. Rather than ship code that cannot be run and therefore cannot be
trusted, the reference implementation is numpy: slower, small enough to verify
against a numerical gradient check, and dependency-free.

THE TRAP THIS MODULE IS BUILT AROUND
------------------------------------
Line art is about 97% white. A network that outputs a blank white page scores
0.97 pixel accuracy and a very low mean-squared error, and has learned nothing
whatsoever. Two decisions follow from that, and they are the substance of this
file rather than details of it:

  * the loss weights ink pixels far above background, so predicting blankness is
    not the cheapest way down;
  * the reported metric is F1 over ink pixels, never accuracy, and every
    evaluation is printed next to the score the blank-page baseline achieves on
    the same data. A result that does not beat that baseline is not a result.
"""

import json
import math
import os

import numpy as np

#: Pixels darker than this in the target are ink; the rest is background.
INK_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

class Corpus:
    """
    A generated corpus, read through its manifest.

    Reads the manifest rather than globbing the directory, because the manifest
    is what says which files belong to the same sample -- and, for the semantic
    pass, what the integer labels mean.
    """

    def __init__(self, root, size=None, limit=None, entries=None):
        self.root = os.path.abspath(root)
        self.size = size
        manifest = os.path.join(self.root, 'manifest.jsonl')
        if entries is not None:
            self.entries = list(entries)
        else:
            if not os.path.isfile(manifest):
                raise FileNotFoundError(manifest)
            self.entries = []
            with open(manifest) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self.entries.append(json.loads(line))
            if limit:
                self.entries = self.entries[:limit]

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return '<Corpus %s samples=%d>' % (self.root, len(self))

    def split(self, which, fraction=0.8):
        """
        Deterministic train/validation split by index.

        By index rather than at random: the split must be identical across runs
        and across processes, or a "validation" score is quietly reporting on
        samples the model has already seen.
        """
        cut = int(len(self.entries) * fraction)
        chosen = self.entries[:cut] if which == 'train' else self.entries[cut:]
        return Corpus(self.root, size=self.size, entries=chosen)

    def load(self, pass_name='rgb', grey=False):
        """Stack one modality into an array of shape (N, C, H, W), in [0, 1]."""
        from PIL import Image
        out = []
        for entry in self.entries:
            rel = entry['files'].get(pass_name)
            if rel is None:
                raise KeyError('sample %s has no %s pass' % (entry['index'], pass_name))
            with Image.open(os.path.join(self.root, rel)) as img:
                img = img.convert('L' if grey else 'RGB')
                if self.size:
                    img = img.resize(self.size, Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0
            if grey:
                arr = arr[None, :, :]
            else:
                arr = arr.transpose(2, 0, 1)
            out.append(arr)
        return np.stack(out) if out else np.zeros((0, 1, 1, 1), dtype=np.float32)

    def vocabulary(self, min_count=2, max_size=None):
        """The proposition vocabulary this corpus can support."""
        import captions
        return captions.vocabulary_from(self.entries, min_count, max_size)[0]

    def coverage(self, vocabulary):
        """Base rate of each proposition, as a fraction of samples."""
        import captions
        return captions.coverage(self.entries, vocabulary)

    def propositions(self, vocabulary):
        """
        Multi-hot proposition targets, shape (N, len(vocabulary)).

        A sample whose manifest carries no facts is all-zero rather than an
        error, so a corpus generated before captions existed still loads -- but
        `has_facts` is how a caller checks, because silently training on a set
        of all-zero targets is exactly the failure this module exists to make
        loud elsewhere.
        """
        import captions
        rows = [captions.encode((e.get('facts') or {}).get('props', []), vocabulary)
                for e in self.entries]
        return np.asarray(rows, dtype=np.float32).reshape(len(self.entries),
                                                          len(vocabulary))

    def captions(self):
        return [(e.get('facts') or {}).get('text', '') for e in self.entries]

    @property
    def has_facts(self):
        return any(e.get('facts') for e in self.entries)

    def pairs(self, source='rgb', target='lineart'):
        """
        (inputs, targets) for training.

        Targets are inverted so that 1 means ink: the network's sigmoid output
        then represents "how much do I believe there is a line here", which is
        the quantity the weighted loss is written in terms of.
        """
        x = self.load(source)
        y = 1.0 - self.load(target, grey=True)
        return x, y



# ---------------------------------------------------------------------------
# propositions
# ---------------------------------------------------------------------------

def proposition_bce(logits, targets, pos_weight=None):
    """
    Multi-label cross-entropy over propositions, with per-class weighting.

    The blank-page trap has an exact analogue here. Most propositions are false
    in most frames, so a model that answers "no" to everything scores high
    accuracy and is useless. `pos_weight` per class is the same remedy the ink
    loss uses, and `suggested_pos_weight` derives it from the corpus rather than
    leaving it as a number to guess.
    """
    probs = sigmoid(logits)
    eps = 1e-7
    p = np.clip(probs, eps, 1 - eps)
    if pos_weight is None:
        pos_weight = np.ones(logits.shape[1], dtype=logits.dtype)
    weight = targets * pos_weight[None, :] + (1.0 - targets)
    loss = -(weight * (targets * np.log(p) + (1 - targets) * np.log(1 - p))).mean()
    grad = (weight * (p - targets)) / targets.size
    return float(loss), grad.astype(logits.dtype)


def suggested_pos_weight(targets, cap=20.0):
    """
    Per-proposition positive weight, from its base rate in the corpus.

    Capped, because a proposition true in one sample of five hundred would
    otherwise get a weight of five hundred and dominate every gradient with a
    class the model has almost no evidence for.
    """
    rate = targets.mean(axis=0)
    rate = np.clip(rate, 1e-6, 1 - 1e-6)
    return np.minimum((1 - rate) / rate, cap).astype(targets.dtype)


def proposition_scores(pred, targets, threshold=0.5):
    """
    Micro and macro F1 over propositions.

    Both, because they fail differently: micro is dominated by the common
    propositions and macro by the rare ones, and a model that has learned
    `scene:tidy` and nothing else scores well on one and badly on the other.
    """
    hard = (pred >= threshold).astype(np.float32)
    truth = (targets >= 0.5).astype(np.float32)

    tp = float((hard * truth).sum())
    fp = float((hard * (1 - truth)).sum())
    fn = float(((1 - hard) * truth).sum())
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if micro_p + micro_r else 0.0)

    per_class = []
    for j in range(truth.shape[1]):
        t = truth[:, j]
        h = hard[:, j]
        ctp = float((h * t).sum())
        cfp = float((h * (1 - t)).sum())
        cfn = float(((1 - h) * t).sum())
        cp = ctp / (ctp + cfp) if ctp + cfp else 0.0
        cr = ctp / (ctp + cfn) if ctp + cfn else 0.0
        per_class.append(2 * cp * cr / (cp + cr) if cp + cr else 0.0)

    return {'micro_f1': micro_f1, 'micro_precision': micro_p,
            'micro_recall': micro_r,
            'macro_f1': float(np.mean(per_class)) if per_class else 0.0,
            'exact_match': float((hard == truth).all(axis=1).mean()),
            'per_class_f1': per_class}


def constant_baseline(train_targets, targets):
    """
    The score from ignoring the image entirely.

    Predicts each proposition's majority class as seen in training. This is the
    blank page of the caption head, and every reported caption score should be
    printed next to it: on a corpus where four fifths of frames contain a mug,
    "yes, there is a mug" is a strong-looking model that has learned nothing.
    """
    majority = (train_targets.mean(axis=0) >= 0.5).astype(np.float32)
    pred = np.repeat(majority[None, :], len(targets), axis=0)
    return proposition_scores(pred, targets, threshold=0.5)


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------

def im2col(x, k, pad, dilation=1):
    """
    (N,C,H,W) -> (N, C*k*k, H*W) with zero padding, for convolution as matmul.

    `dilation` spaces the taps apart, which widens the receptive field without
    adding parameters or losing resolution -- the cheap way to let a pixel see
    further when the answer depends on context rather than on detail.
    """
    n, c, h, w = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.empty((n, c * k * k, h * w), dtype=x.dtype)
    idx = 0
    for dy in range(k):
        for dx in range(k):
            oy, ox = dy * dilation, dx * dilation
            patch = padded[:, :, oy:oy + h, ox:ox + w]
            cols[:, idx:idx + c, :] = patch.reshape(n, c, h * w)
            idx += c
    return cols


def col2im(cols, shape, k, pad, dilation=1):
    """Adjoint of im2col: scatter-add gradients back to image positions."""
    n, c, h, w = shape
    out = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=cols.dtype)
    idx = 0
    for dy in range(k):
        for dx in range(k):
            oy, ox = dy * dilation, dx * dilation
            out[:, :, oy:oy + h, ox:ox + w] += cols[:, idx:idx + c, :].reshape(n, c, h, w)
            idx += c
    return out[:, :, pad:pad + h, pad:pad + w] if pad else out


class Conv2d:
    """A 3x3-style convolution, same padding, implemented as one matmul."""

    def __init__(self, in_ch, out_ch, k=3, rng=None, dtype=np.float32, dilation=1):
        rng = rng or np.random.default_rng(0)
        ## He initialisation: with ReLU throwing away half the signal, scaling
        ## by sqrt(2/fan_in) keeps activations from collapsing as depth grows.
        fan_in = in_ch * k * k
        ## dtype is exposed so a gradient check can run in float64: a numerical
        ## derivative is a difference of two nearly equal numbers, and in
        ## float32 the cancellation swamps the answer -- the check fails on
        ## arithmetic rather than on a wrong gradient.
        self.dtype = dtype
        self.w = (rng.standard_normal((out_ch, fan_in)) * math.sqrt(2.0 / fan_in)).astype(dtype)
        self.b = np.zeros(out_ch, dtype=dtype)
        self.k = k
        self.dilation = dilation
        ## 'Same' padding must grow with the dilation, or a dilated layer
        ## silently crops the image and every later layer is misaligned.
        self.pad = (k // 2) * dilation
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.cache = None
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def params(self):
        return ((self.w, self.dw), (self.b, self.db))

    def forward(self, x):
        n, c, h, w = x.shape
        cols = im2col(x, self.k, self.pad, self.dilation)   # (N, C*k*k, HW)
        out = np.einsum('of,nfp->nop', self.w, cols) + self.b[None, :, None]
        self.cache = (x.shape, cols)
        return out.reshape(n, self.out_ch, h, w)

    def backward(self, grad):
        shape, cols = self.cache
        n, c, h, w = shape
        g = grad.reshape(n, self.out_ch, h * w)
        self.dw[...] = np.einsum('nop,nfp->of', g, cols)
        self.db[...] = g.sum(axis=(0, 2))
        dcols = np.einsum('of,nop->nfp', self.w, g)
        return col2im(dcols, shape, self.k, self.pad, self.dilation)


class ReLU:
    def __init__(self):
        self.mask = None

    def params(self):
        return ()

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask



class GlobalPool:
    """
    Mean over the spatial dimensions: (N, C, H, W) -> (N, C).

    Mean rather than max. A proposition like `scene:disordered` is a statement
    about the whole frame, and max-pooling would let one strongly activating
    pixel carry it -- which is how a detector learns to fire on a highlight
    that happened to co-occur with fallen objects in training.
    """

    def __init__(self):
        self.shape = None

    def params(self):
        return ()

    def forward(self, x):
        self.shape = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, grad):
        n, c, h, w = self.shape
        ## Each input pixel contributed 1/(H*W) of the mean.
        return np.repeat(np.repeat(grad[:, :, None, None], h, axis=2), w, axis=3) / (h * w)


class BandPool:
    """
    Mean over height, but into `bins` columns across the width: (N,C,H,W) -> (N,C*bins).

    `GlobalPool` collapses both spatial axes, which is right for a question
    about the whole frame -- is anything tipped over -- and wrong for a question
    about direction. A steering command depends on *where* the goal is
    left-to-right, and a mean over width destroys that by construction: on the
    control corpus the goal's horizontal centroid correlates -0.77 with the
    expert's commanded yaw rate, while the channel mean correlates -0.05.

    Keeping a handful of columns preserves the horizontal layout at a fraction
    of the parameters a flatten would cost. Height is still collapsed, because
    for a camera on a driving robot the vertical axis mostly encodes distance,
    which the other channels already carry.
    """

    def __init__(self, bins=8):
        self.bins = bins
        self.shape = None
        self.edges = None

    def params(self):
        return ()

    def spans(self, width):
        """
        Column range for each bin, guaranteed non-empty.

        When the feature map is narrower than the bin count -- which happens on
        small inputs, and in tests -- evenly spaced edges produce empty bins,
        and the mean of an empty slice is NaN. Widening those bins to one column
        keeps every bin defined and keeps the output width fixed, which matters
        because the following Linear layer's shape is decided at construction
        and cannot change with the input.
        """
        out = []
        for b in range(self.bins):
            lo = (b * width) // self.bins
            hi = ((b + 1) * width) // self.bins
            if hi <= lo:
                ## Overlap rather than emit nothing; adjacent bins then share a
                ## column, which is the honest answer when the map is narrower
                ## than the number of questions being asked of it.
                lo = min(lo, max(0, width - 1))
                hi = lo + 1
            out.append((lo, hi))
        return out

    def forward(self, x):
        n, c, h, w = x.shape
        self.shape = x.shape
        self.edges = self.spans(w)
        out = np.empty((n, c * self.bins), dtype=x.dtype)
        for b, (lo, hi) in enumerate(self.edges):
            out[:, b * c:(b + 1) * c] = x[:, :, :, lo:hi].mean(axis=(2, 3))
        return out

    def backward(self, grad):
        n, c, h, w = self.shape
        out = np.zeros(self.shape, dtype=grad.dtype)
        for b, (lo, hi) in enumerate(self.edges):
            share = grad[:, b * c:(b + 1) * c] / float(h * (hi - lo))
            ## Accumulate, not assign: bins overlap when the map is narrow, and
            ## assigning would silently drop one bin's gradient.
            out[:, :, :, lo:hi] += share[:, :, None, None]
        return out


class Linear:
    """A fully-connected layer: (N, in) -> (N, out)."""

    def __init__(self, in_features, out_features, rng=None, dtype=np.float32):
        rng = rng or np.random.default_rng(0)
        self.dtype = dtype
        self.w = (rng.standard_normal((out_features, in_features))
                  * math.sqrt(2.0 / in_features)).astype(dtype)
        self.b = np.zeros(out_features, dtype=dtype)
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.x = None

    def params(self):
        return ((self.w, self.dw), (self.b, self.db))

    def forward(self, x):
        self.x = x
        return x @ self.w.T + self.b

    def backward(self, grad):
        self.dw[...] = grad.T @ self.x
        self.db[...] = grad.sum(axis=0)
        return grad @ self.w


class MultiModalNet:
    """
    One trunk, two heads: line art per pixel, propositions per frame.

    The trunk is the same convolutional stack `LineArtNet` uses, and the line
    art head is unchanged, so a multi-modal net and a line-art-only net are
    the same model plus one extra output. That is the point of training them
    together: the caption head is supervision the corpus was already carrying
    for free, and if predicting `mug:tipped` requires features that also help
    localise a contour, both tasks benefit.

    The two heads are separate branches over a shared trunk, so their gradients
    are summed into it. `loss_weight` sets how loudly the caption head speaks;
    at 0 this is exactly `LineArtNet` with a dead branch attached.
    """

    def __init__(self, n_props, width=12, depth=3, seed=0, in_ch=3,
                 dtype=np.float32, dilations=None):
        rng = np.random.default_rng(seed)
        self.dtype = dtype
        self.n_props = n_props
        self.trunk = []
        channels = in_ch
        hidden = max(1, depth - 1)
        if dilations is None:
            dilations = [1] * hidden
        dilations = list(dilations)[:hidden] + [1] * max(0, hidden - len(dilations))
        self.dilations = dilations
        for d in dilations:
            self.trunk.append(Conv2d(channels, width, rng=rng, dtype=dtype, dilation=d))
            self.trunk.append(ReLU())
            channels = width
        self.ink_head = Conv2d(channels, 1, rng=rng, dtype=dtype)
        self.pool = GlobalPool()
        self.prop_head = Linear(channels, max(1, n_props), rng=rng, dtype=dtype)

    def params(self):
        out = []
        for layer in self.trunk:
            out.extend(layer.params())
        out.extend(self.ink_head.params())
        out.extend(self.prop_head.params())
        return out

    def forward(self, x):
        """Returns (ink logits (N,1,H,W), proposition logits (N,P))."""
        for layer in self.trunk:
            x = layer.forward(x)
        self.features = x
        ink = self.ink_head.forward(x)
        props = self.prop_head.forward(self.pool.forward(x))
        return ink, props

    def backward(self, d_ink, d_props):
        """
        Push both heads' gradients back through the shared trunk.

        Summed, not run twice: the trunk's stored activations belong to one
        forward pass, and backpropagating through it a second time would
        overwrite the first head's parameter gradients before the optimiser saw
        them, because Conv2d assigns to `dw` rather than accumulating.
        """
        grad = self.ink_head.backward(d_ink)
        if d_props is not None:
            grad = grad + self.pool.backward(self.prop_head.backward(d_props))
        for layer in reversed(self.trunk):
            grad = layer.backward(grad)
        return grad

    def predict(self, x):
        ink, props = self.forward(x)
        return sigmoid(ink), sigmoid(props)

    def save(self, path):
        arrays = {}
        for i, (p, _g) in enumerate(self.params()):
            arrays['p%d' % i] = p
        np.savez(path, **arrays)
        return path

    def load(self, path):
        data = np.load(path)
        for i, (p, _g) in enumerate(self.params()):
            p[...] = data['p%d' % i]
        return self


class LineArtNet:
    """
    A small fully-convolutional network: photograph in, ink probability out.

    No pooling and no strides. Line art is a local, high-frequency function of
    the image -- an edge is visible in its own neighbourhood -- so resolution is
    worth more here than receptive field, and keeping it avoids the checkerboard
    artefacts that upsampling introduces on thin strokes.
    """

    def __init__(self, width=12, depth=3, seed=0, in_ch=3, dtype=np.float32,
                 dilations=None):
        rng = np.random.default_rng(seed)
        self.dtype = dtype
        self.layers = []
        channels = in_ch
        ## Dilations widen the receptive field geometrically instead of
        ## linearly. Five plain 3x3 layers see 11 pixels; the same five with
        ## dilations 1,2,4,8 see 63, which is most of a 64-wide image -- and
        ## whether an edge is an object boundary is a question about context,
        ## not about the pixel.
        hidden = max(1, depth - 1)
        if dilations is None:
            dilations = [1] * hidden
        dilations = list(dilations)[:hidden] + [1] * max(0, hidden - len(dilations))
        self.dilations = dilations
        for d in dilations:
            self.layers.append(Conv2d(channels, width, rng=rng, dtype=dtype, dilation=d))
            self.layers.append(ReLU())
            channels = width
        ## Final layer emits one logit per pixel; the sigmoid lives in the loss,
        ## where it can be fused with the log for numerical stability.
        self.head = Conv2d(channels, 1, rng=rng, dtype=dtype)
        self.layers.append(self.head)

    def params(self):
        out = []
        for layer in self.layers:
            out.extend(layer.params())
        return out

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict(self, x):
        """Ink probability in [0, 1]."""
        return sigmoid(self.forward(x))

    def state(self):
        return [(p.copy(), None) for p, _g in self.params()]

    def save(self, path):
        arrays = {}
        for i, (p, _g) in enumerate(self.params()):
            arrays['p%d' % i] = p
        np.savez(path, **arrays)
        return path

    def load(self, path):
        data = np.load(path)
        for i, (p, _g) in enumerate(self.params()):
            p[...] = data['p%d' % i]
        return self


def sigmoid(z):
    ## Branch on sign rather than exp(-z) everywhere: exp overflows for large
    ## negative z, which is exactly where a confident background prediction sits.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

def weighted_bce(logits, targets, ink_weight):
    """
    Binary cross-entropy with ink pixels weighted up, and its gradient.

    Computed from the logits with the log-sum-exp form rather than from a
    sigmoid output, because log(0) is where the naive version produces NaN --
    and with a 97% background class it will reach that point quickly.

    `ink_weight` is what stops the blank page being the cheapest answer.
    """
    weights = np.where(targets > 0.5, ink_weight, 1.0).astype(logits.dtype)
    ## log(1+exp(-|z|)) + max(z,0) - z*t  is the stable form of the BCE.
    loss_per_px = (np.maximum(logits, 0) - logits * targets
                   + np.log1p(np.exp(-np.abs(logits))))
    total = float((loss_per_px * weights).sum() / weights.sum())
    grad = weights * (sigmoid(logits) - targets) / weights.sum()
    return total, grad.astype(logits.dtype)


def ink_fraction(targets):
    return float((targets > 0.5).mean())


def suggested_ink_weight(targets, cap=40.0):
    """
    Weight that makes ink and background contribute comparably.

    Capped: the raw ratio on a sparse drawing is in the hundreds, and at that
    level the network minimises loss by covering the page in ink instead, which
    is the same failure wearing the opposite coat.
    """
    ink = ink_fraction(targets)
    if ink <= 0:
        return 1.0
    return float(min(cap, (1.0 - ink) / ink))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def dilate(mask, radius=1):
    """
    Grow a boolean mask by `radius` pixels, without scipy.

    Implemented as shifted ORs: for the radii used here (one or two pixels) that
    is a handful of array operations, and it keeps this module dependency-free.
    """
    if radius <= 0:
        return mask
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(mask, dy, axis=-2), dx, axis=-1)
            ## Rolled-in edges wrap around, which would match a stroke on the
            ## opposite side of the image. Blank the wrapped band.
            if dy > 0:
                shifted[..., :dy, :] = False
            elif dy < 0:
                shifted[..., dy:, :] = False
            if dx > 0:
                shifted[..., :, :dx] = False
            elif dx < 0:
                shifted[..., :, dx:] = False
            out |= shifted
    return out


def scores(pred, targets, threshold=0.5, tolerance=0):
    """
    Precision, recall and F1 over ink pixels.

    `tolerance` allows a match within that many pixels, which is how boundary
    detection is normally scored. The reason is specific to this target: strokes
    are one pixel wide, so a prediction that traces a contour perfectly but one
    pixel to the left scores *zero* on both precision and recall for that
    contour. At tolerance 0 the metric is measuring localisation as much as
    detection, and the two are worth separating before concluding a model cannot
    see edges.

    Strict (tolerance 0) remains the default, because a tolerant score is easy
    to quote without the qualifier and should never be the headline by accident.
    """
    """
    Precision, recall and F1 over ink pixels, plus accuracy for contrast.

    Accuracy is reported only so it can be seen doing its misleading work: on a
    97% white image a blank prediction scores 0.97 while finding nothing.
    """
    p = pred > threshold
    t = targets > 0.5
    if tolerance:
        ## A prediction counts if a true stroke is near it, and a true stroke
        ## counts if a prediction is near it. Scored against separately dilated
        ## masks rather than one, or a thick blob would earn recall it has not
        ## demonstrated.
        near_true = dilate(t, tolerance)
        near_pred = dilate(p, tolerance)
        tp_p = float(np.sum(p & near_true))
        tp_r = float(np.sum(t & near_pred))
        fp = float(np.sum(p & ~near_true))
        fn = float(np.sum(t & ~near_pred))
        precision = tp_p / (tp_p + fp) if tp_p + fp else 0.0
        recall = tp_r / (tp_r + fn) if tp_r + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {'precision': precision, 'recall': recall, 'f1': f1,
                'accuracy': float(np.mean(p == t)), 'ink_fraction': ink_fraction(targets),
                'tolerance': tolerance}
    tp = float(np.sum(p & t))
    fp = float(np.sum(p & ~t))
    fn = float(np.sum(~p & t))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': float(np.mean(p == t)),
        'ink_fraction': ink_fraction(targets),
    }


def blank_baseline(targets):
    """
    What a network that learned nothing would score.

    Every result must be read against this. A model that cannot beat the blank
    page has not learned to see edges; it has learned that paper is white.
    """
    pred = np.zeros_like(targets)
    return scores(pred, targets)


def best_threshold(net, x, y, candidates=None, batch=8):
    """
    Threshold maximising F1, chosen on whichever data is passed in.

    Worth doing because the network is systematically under-confident here: with
    ink outnumbered a hundred to one, even a weighted loss leaves most true ink
    pixels below 0.5, so the default threshold throws away recall the model
    actually has. Choose this on *training* data and report on validation --
    picking it on the validation set is how a tuned threshold turns into an
    inflated score.
    """
    preds = []
    for i in range(0, len(x), batch):
        preds.append(net.predict(x[i:i + batch]))
    pred = np.concatenate(preds) if preds else np.zeros_like(y)
    best, best_f1 = 0.5, -1.0
    for t in (candidates if candidates is not None
              else [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]):
        f1 = scores(pred, y, t)['f1']
        if f1 > best_f1:
            best, best_f1 = t, f1
    return best, best_f1


def evaluate(net, x, y, threshold=0.5, batch=8, tolerance=0):
    preds = []
    for i in range(0, len(x), batch):
        preds.append(net.predict(x[i:i + batch]))
    pred = np.concatenate(preds) if preds else np.zeros_like(y)
    result = scores(pred, y, threshold, tolerance)
    result['baseline_f1'] = blank_baseline(y)['f1']
    return result


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

class Adam:
    """Adam, because a hand-tuned SGD schedule is one more thing to get wrong."""

    def __init__(self, params, lr=3e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p, _g in self.params]
        self.v = [np.zeros_like(p) for p, _g in self.params]

    def step(self):
        self.t += 1
        correction1 = 1 - self.b1 ** self.t
        correction2 = 1 - self.b2 ** self.t
        for i, (p, g) in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            p -= self.lr * (self.m[i] / correction1) / (
                np.sqrt(self.v[i] / correction2) + self.eps)


def train(net, x, y, epochs=10, batch=4, lr=3e-3, ink_weight=None, seed=0,
          log=print, val=None):
    """
    Fit the network, reporting F1 against the blank-page baseline each epoch.

    Reported rather than merely computed: a falling loss on this task means very
    little on its own, because the loss falls fastest when the network discovers
    that most of the page is white.
    """
    if ink_weight is None:
        ink_weight = suggested_ink_weight(y)
    opt = Adam(net.params(), lr=lr)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        total = 0.0
        batches = 0
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            logits = net.forward(x[idx])
            loss, grad = weighted_bce(logits, y[idx], ink_weight)
            net.backward(grad)
            opt.step()
            total += loss
            batches += 1
        entry = {'epoch': epoch, 'loss': total / max(1, batches)}
        entry.update({'train_' + k: v for k, v in evaluate(net, x, y).items()})
        if val is not None:
            entry.update({'val_' + k: v for k, v in evaluate(net, *val).items()})
        history.append(entry)
        if log:
            message = 'epoch %2d  loss %.4f  train F1 %.3f' % (
                entry['epoch'], entry['loss'], entry['train_f1'])
            if val is not None:
                message += '  val F1 %.3f (blank %.3f)' % (
                    entry['val_f1'], entry['val_baseline_f1'])
            log(message)
    return history


def evaluate_multimodal(net, x, y_ink, y_props, threshold=0.5, batch=8,
                        tolerance=0):
    """Both heads, scored on the same forward pass."""
    inks, props = [], []
    for start in range(0, len(x), batch):
        ink, prop = net.predict(x[start:start + batch])
        inks.append(ink)
        props.append(prop)
    ink = np.concatenate(inks) if inks else np.zeros_like(y_ink)
    prop = np.concatenate(props) if props else np.zeros_like(y_props)

    out = scores(ink, y_ink, threshold=threshold, tolerance=tolerance)
    ## The blank-page score, carried alongside as `evaluate` does. Every ink
    ## number in this module is reported next to it, and an evaluation function
    ## that omits it invites exactly the comparison it exists to prevent.
    out['baseline_f1'] = blank_baseline(y_ink)['f1']
    out.update({('prop_' + k): v
                for k, v in proposition_scores(prop, y_props, threshold).items()
                if k != 'per_class_f1'})
    return out


def train_multimodal(net, x, y_ink, y_props, epochs=10, batch=4, lr=3e-3,
                     ink_weight=None, pos_weight=None, caption_weight=1.0,
                     seed=0, log=print, val=None):
    """
    Fit both heads at once.

    `caption_weight` scales the proposition loss before it enters the shared
    trunk. It is exposed rather than fixed because the two losses are not
    commensurate: the ink loss is a mean over tens of thousands of pixels and
    the caption loss a mean over a few dozen propositions, so equal weight is
    not equal influence, and the right ratio is an empirical question about a
    particular corpus rather than a constant.

    Every epoch reports both heads against their own do-nothing baseline -- the
    blank page for ink, the majority class for propositions. A caption score
    printed without its baseline is not interpretable.
    """
    if ink_weight is None:
        ink_weight = suggested_ink_weight(y_ink)
    if pos_weight is None:
        pos_weight = suggested_pos_weight(y_props)
    opt = Adam(net.params(), lr=lr)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        totals = [0.0, 0.0]
        batches = 0
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            ink_logits, prop_logits = net.forward(x[idx])
            ink_loss, d_ink = weighted_bce(ink_logits, y_ink[idx], ink_weight)
            prop_loss, d_prop = proposition_bce(prop_logits, y_props[idx],
                                                pos_weight)
            net.backward(d_ink, d_prop * caption_weight)
            opt.step()
            totals[0] += ink_loss
            totals[1] += prop_loss
            batches += 1

        entry = {'epoch': epoch,
                 'ink_loss': totals[0] / max(1, batches),
                 'caption_loss': totals[1] / max(1, batches)}
        entry.update({'train_' + k: v for k, v in
                      evaluate_multimodal(net, x, y_ink, y_props).items()})
        if val is not None:
            vx, vy_ink, vy_props = val
            entry.update({'val_' + k: v for k, v in
                          evaluate_multimodal(net, vx, vy_ink, vy_props).items()})
            entry['val_prop_baseline_f1'] = constant_baseline(
                y_props, vy_props)['micro_f1']
        history.append(entry)
        if log:
            message = ('epoch %2d  ink %.4f / caption %.4f  '
                       'train F1 %.3f  caption F1 %.3f'
                       % (entry['epoch'], entry['ink_loss'],
                          entry['caption_loss'], entry['train_f1'],
                          entry['train_prop_micro_f1']))
            if val is not None:
                message += ('  |  val F1 %.3f (blank %.3f)  '
                            'caption %.3f (majority %.3f)'
                            % (entry['val_f1'], entry['val_baseline_f1'],
                               entry['val_prop_micro_f1'],
                               entry['val_prop_baseline_f1']))
            log(message)
    return history
