"""
Writing the corpus: aligned samples, and a manifest that explains them.

A directory of PNGs is not a dataset. What makes it one is knowing, for every
sample, which files belong together, what the pixel values mean, and how to get
the sample back if it looks wrong. This module writes all three.

    ds = Dataset('/tmp/corpus', passes=('rgb', 'depth', 'segmentation', 'lineart'))
    ds.label('obstacle', 7)
    ds.write(index, capture, lineart=path, seed=seed, meta={...})
    ds.close()

THE MANIFEST IS THE POINT
-------------------------
Each sample gets one JSON line recording its files, its seed, the camera, and
the label map in force when it was written. That last item is what turns the
ObjectID pass from an image into a semantic map: pixel value 7 means nothing on
its own, and means "obstacle" only because the manifest says so for this
sample. Storing the map per sample rather than once per corpus means a corpus
whose labelling changed halfway through is still readable, instead of being
silently mislabelled from the point of the change.

The seed is stored for the same reason: a sample that looks wrong during
training can be regenerated exactly, on its own, without rerunning the corpus.

JSON Lines rather than one JSON document, because a generation run that dies at
sample 40,000 should leave 40,000 usable samples rather than an unterminated
array.

ALIGNMENT IS CHECKED, NOT ASSUMED
---------------------------------
The premise of the whole pipeline is that the modalities are pixel-aligned.
`verify()` re-opens what was written and confirms every sample's files exist and
agree on resolution. Misalignment is the one corruption that trains a network to
be confidently wrong, and it is invisible in any single image.
"""

import json
import os
import shutil


class Dataset:
    """
    Accumulates samples on disk and writes a manifest describing them.

    Files are named `<split>/<index:06d>.<pass>.<ext>`, which sorts in
    generation order and groups a sample's modalities adjacently.
    """

    MANIFEST = 'manifest.jsonl'
    INFO = 'dataset.json'

    def __init__(self, root, passes=('rgb', 'depth', 'segmentation'), split='train',
                 labels=None, overwrite=False, note='', manifest=None):
        self.root = os.path.abspath(root)
        self.split = split
        self.passes = tuple(passes)
        self.labels = dict(labels or {})
        self.note = note
        self.count = 0
        self._handle = None
        ## Shards write separate manifests and are merged afterwards. Appending
        ## to one file from several processes would interleave partial lines:
        ## the writes are small, but "small enough that it probably works" is
        ## not a property a corpus should depend on.
        self.manifest_name = manifest or self.MANIFEST
        if overwrite and os.path.isdir(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.sample_dir, exist_ok=True)

    def __repr__(self):
        return '<Dataset %s split=%s samples=%d>' % (self.root, self.split, self.count)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @property
    def sample_dir(self):
        return os.path.join(self.root, self.split)

    @property
    def manifest_path(self):
        return os.path.join(self.root, self.manifest_name)

    # -- labels -------------------------------------------------------------

    def label(self, token, index):
        """
        Bind a semantic token to an ObjectID index.

        This is the translation layer: the segmentation pass stores integers,
        and a corpus is only usable if something records what those integers
        mean. Written into every sample's manifest entry, not just the header.
        """
        self.labels[str(token)] = int(index)
        return self

    def labels_by_index(self):
        return {index: token for token, index in self.labels.items()}

    # -- writing ------------------------------------------------------------

    def path_for(self, index, pass_name, ext):
        return os.path.join(self.sample_dir, '%06d.%s%s' % (index, pass_name, ext))

    def write(self, index, capture=None, seed=None, meta=None, facts=None,
              **extra):
        """
        Record one sample.

        `capture` is the dict robotsim's SensorRig returns for one camera --
        {pass: path} -- and `extra` takes any further modality by keyword, which
        is how line art (rendered separately) joins the same sample. `facts` is
        the scene description: propositions and a caption, which are data rather
        than a file and so are stored inline.

        Files already in place are moved into the dataset rather than copied,
        so generation does not pay twice for every frame it writes.
        """
        files = {}
        sources = dict(capture or {})
        sources.update({k: v for k, v in extra.items() if v})
        for pass_name, source in sources.items():
            if not source or not os.path.isfile(source):
                raise FileNotFoundError('%s pass missing for sample %d: %s'
                                        % (pass_name, index, source))
            ext = os.path.splitext(source)[1] or '.png'
            target = self.path_for(index, pass_name, ext)
            if os.path.abspath(source) != target:
                shutil.move(source, target)
            files[pass_name] = os.path.relpath(target, self.root)

        missing = [p for p in self.passes if p not in files]
        if missing:
            ## Refuse rather than write a partial sample: a corpus where some
            ## samples silently lack a modality produces a training loader that
            ## either crashes late or, worse, skips them and biases the set.
            raise ValueError('sample %d is missing %s' % (index, ', '.join(missing)))

        entry = {
            'index': index,
            'split': self.split,
            'files': files,
            'labels': dict(self.labels),
        }
        if seed is not None:
            entry['seed'] = seed
        if meta:
            entry['meta'] = meta
        if facts:
            ## Propositions and a caption live in the manifest rather than in a
            ## sidecar file: they are small, they are per-sample, and keeping
            ## them beside `labels` means one read gives a loader everything it
            ## needs to build a supervision target.
            entry['facts'] = facts

        if self._handle is None:
            self._handle = open(self.manifest_path, 'a')
        self._handle.write(json.dumps(entry, sort_keys=True) + '\n')
        ## Flushed per sample: a run killed mid-generation should leave every
        ## completed sample readable.
        self._handle.flush()
        self.count += 1
        return entry

    def prune(self, indices):
        """
        Drop samples from the manifest, leaving their files on disk.

        Randomised generation occasionally produces a sample with nothing in it
        -- a viewpoint that happened to face empty sky, a lighting draw that
        washed the frame flat. One such sample should not invalidate the other
        four hundred, and deleting it by hand is how a corpus quietly stops
        matching its manifest. The files are left alone so a dropped sample can
        still be looked at to understand why it was dropped.
        """
        indices = set(indices)
        if not indices:
            return 0
        kept = [e for e in self.entries() if e['index'] not in indices]
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        with open(self.manifest_path, 'w') as fh:
            for entry in kept:
                fh.write(json.dumps(entry, sort_keys=True) + '\n')
        self.count = len(kept)
        return len(indices)

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        info = {
            'passes': list(self.passes),
            'labels': self.labels,
            'samples': self.count,
            'split': self.split,
            'note': self.note,
        }
        with open(os.path.join(self.root, self.INFO), 'w') as fh:
            json.dump(info, fh, indent=2, sort_keys=True)
        return self

    # -- reading back -------------------------------------------------------

    def entries(self):
        """Every manifest entry, in written order."""
        if not os.path.isfile(self.manifest_path):
            return []
        out = []
        with open(self.manifest_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def exposure(self, path, buckets=32):
        """
        Crude histogram spread of an image, in [0, 1].

        Near zero means every pixel is essentially the same value: a black
        frame, a blown-out one, or a render whose lighting silently failed.
        Cheap enough to run over a whole corpus.
        """
        try:
            from PIL import Image
        except ImportError:
            return None
        try:
            with Image.open(path) as img:
                small = img.convert('L').resize((64, 64))
                values = list(small.getdata())
        except Exception:
            return None
        if not values:
            return None
        seen = set(v * buckets // 256 for v in values)
        return len(seen) / float(buckets)

    def verify(self, check_size=True, check_exposure=True, min_spread=0.09):
        """
        Re-open the corpus and check it is what the manifest claims.

        Returns a list of problems, empty when the corpus is sound. Alignment is
        the thing being protected: modalities that disagree on resolution are
        not aligned, and a network trained on them learns a systematic offset it
        can never recover from.
        """
        problems = []
        for entry in self.entries():
            index = entry['index']
            sizes = {}
            for pass_name in self.passes:
                rel = entry['files'].get(pass_name)
                if rel is None:
                    problems.append('sample %d: no %s' % (index, pass_name))
                    continue
                path = os.path.join(self.root, rel)
                if not os.path.isfile(path):
                    problems.append('sample %d: missing file %s' % (index, rel))
                    continue
                if check_size:
                    size = _image_size(path)
                    if size:
                        sizes[pass_name] = size
            if check_size and len(set(sizes.values())) > 1:
                problems.append('sample %d: modalities disagree on size: %s'
                                % (index, sizes))
            if check_exposure and 'rgb' in entry['files']:
                ## An unlit render is the corruption that looks like success:
                ## the file exists, the resolution matches, the manifest is
                ## complete, and every pixel is black. Only looking at the
                ## pixels catches it, so verification looks at the pixels.
                path = os.path.join(self.root, entry['files']['rgb'])
                spread = self.exposure(path)
                if spread is not None and spread < min_spread:
                    problems.append('sample %d: rgb has almost no tonal range '
                                    '(spread %.3f) -- unlit or blown out'
                                    % (index, spread))
        return problems


def indices_in(problems):
    """Sample indices mentioned by verify(), for pruning."""
    import re
    out = set()
    for text in problems:
        match = re.search(r'sample (\d+)', text)
        if match:
            out.add(int(match.group(1)))
    return out


def merge_manifests(root, pattern='manifest.*.jsonl', out=None, remove=True):
    """
    Combine shard manifests into one, ordered by sample index.

    Sorted by index rather than by shard, so the deterministic train/validation
    split by position means the same thing however many workers produced the
    corpus -- otherwise the split silently changes with the worker count, and
    two runs that should be comparable are not.
    """
    import glob
    out = out or os.path.join(root, Dataset.MANIFEST)
    shards = [p for p in sorted(glob.glob(os.path.join(root, pattern)))
              if os.path.abspath(p) != os.path.abspath(out)]
    if not shards:
        ## Nothing to merge. Returning without writing matters: an unsharded run
        ## has already written the real manifest, and truncating it here would
        ## leave a corpus whose files all exist and whose index is empty.
        return len(read_manifest(out)) if os.path.isfile(out) else 0
    entries = []
    for path in shards:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    entries.sort(key=lambda e: e['index'])
    with open(out, 'w') as fh:
        for entry in entries:
            fh.write(json.dumps(entry, sort_keys=True) + '\n')
    if remove:
        for path in shards:
            os.remove(path)
    return len(entries)


def read_manifest(path):
    """Every entry in a manifest file, or [] if it is not there."""
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _image_size(path):
    """
    (width, height) for a written pass, or None if it cannot be read.

    PNG is read from the header directly and EXR through OpenImageIO-free
    means, because the point is to check alignment cheaply across a whole
    corpus -- decoding every float EXR to verify a number in its header would
    make verification cost more than generation.
    """
    try:
        with open(path, 'rb') as fh:
            head = fh.read(64)
    except OSError:
        return None
    if head[:8] == b'\x89PNG\r\n\x1a\n':
        import struct
        width, height = struct.unpack('>II', head[16:24])
        return (width, height)
    if head[:4] == b'\x76\x2f\x31\x01':
        ## OpenEXR: the data window lives in the header, but parsing it
        ## properly means walking typed attributes. Pillow reads it if present.
        try:
            from PIL import Image
            with Image.open(path) as img:
                return img.size
        except Exception:
            return None
    return None
