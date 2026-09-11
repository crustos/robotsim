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
# layers
# ---------------------------------------------------------------------------

def im2col(x, k, pad):
    """(N,C,H,W) -> (N, C*k*k, H*W) with zero padding, for convolution as matmul."""
    n, c, h, w = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.empty((n, c * k * k, h * w), dtype=x.dtype)
    idx = 0
    for dy in range(k):
        for dx in range(k):
            patch = padded[:, :, dy:dy + h, dx:dx + w]
            cols[:, idx:idx + c, :] = patch.reshape(n, c, h * w)
            idx += c
    return cols


def col2im(cols, shape, k, pad):
    """Adjoint of im2col: scatter-add gradients back to image positions."""
    n, c, h, w = shape
    out = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=cols.dtype)
    idx = 0
    for dy in range(k):
        for dx in range(k):
            out[:, :, dy:dy + h, dx:dx + w] += cols[:, idx:idx + c, :].reshape(n, c, h, w)
            idx += c
    return out[:, :, pad:pad + h, pad:pad + w] if pad else out


class Conv2d:
    """A 3x3-style convolution, same padding, implemented as one matmul."""

    def __init__(self, in_ch, out_ch, k=3, rng=None, dtype=np.float32):
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
        self.pad = k // 2
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.cache = None
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def params(self):
        return ((self.w, self.dw), (self.b, self.db))

    def forward(self, x):
        n, c, h, w = x.shape
        cols = im2col(x, self.k, self.pad)                  # (N, C*k*k, HW)
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
        return col2im(dcols, shape, self.k, self.pad)


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


class LineArtNet:
    """
    A small fully-convolutional network: photograph in, ink probability out.

    No pooling and no strides. Line art is a local, high-frequency function of
    the image -- an edge is visible in its own neighbourhood -- so resolution is
    worth more here than receptive field, and keeping it avoids the checkerboard
    artefacts that upsampling introduces on thin strokes.
    """

    def __init__(self, width=12, depth=3, seed=0, in_ch=3, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.dtype = dtype
        self.layers = []
        channels = in_ch
        for _ in range(max(1, depth - 1)):
            self.layers.append(Conv2d(channels, width, rng=rng, dtype=dtype))
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

def scores(pred, targets, threshold=0.5):
    """
    Precision, recall and F1 over ink pixels, plus accuracy for contrast.

    Accuracy is reported only so it can be seen doing its misleading work: on a
    97% white image a blank prediction scores 0.97 while finding nothing.
    """
    p = pred > threshold
    t = targets > 0.5
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


def evaluate(net, x, y, threshold=0.5, batch=8):
    preds = []
    for i in range(0, len(x), batch):
        preds.append(net.predict(x[i:i + batch]))
    pred = np.concatenate(preds) if preds else np.zeros_like(y)
    result = scores(pred, y, threshold)
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
