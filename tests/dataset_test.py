#!../headless.py
print('hello dataset test...')
import bpy, os, math, json
import npr, randomize, dataset
from dataset import Dataset
from randomize import Randomizer
from npr import LineArt

OUT = '/tmp/robotsim-dataset-test'
os.system('rm -rf %s' % OUT)
W, H = 96, 72


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


def fresh_scene():
    ground = create_cube('GROUND', size=(200, 200, 0.2), location=(0, 0, -0.1))
    ground.pass_index = SEGMENT_CLASSES['ground']
    mat = bpy.data.materials.new('GND')
    mat.use_nodes = True
    ground.data.materials.append(mat)
    bpy.context.view_layer.update()
    return ground


def test_randomizer_is_deterministic():
    """The same seed must rebuild the same scene, or a corpus is unreproducible."""
    def layout(seed):
        r = Randomizer(seed=seed, extent=8.0)
        objs = r.scene(obstacles=(5, 5))
        out = [tuple(round(v, 6) for v in o.location) for o in objs]
        r.clear()
        return out

    a, b, c = layout(42), layout(42), layout(43)
    print('seed 42: %d objects, seed 43: %d objects' % (len(a), len(c)))
    assert a == b, 'same seed gave a different scene'
    assert a != c, 'different seeds gave the same scene'

    ## reseed rewinds the same generator
    r = Randomizer(seed=7)
    first = [tuple(round(v, 6) for v in o.location) for o in r.scene(obstacles=(4, 4))]
    r.reseed(7)
    again = [tuple(round(v, 6) for v in o.location) for o in r.scene(obstacles=(4, 4))]
    r.clear()
    assert first == again, 'reseed did not rewind'
    print('determinism OK')


def test_randomizer_clears_only_its_own():
    """clear() must not take the scene with it."""
    ground = fresh_scene()
    bot = Robot(arms=[], cameras='none')
    r = Randomizer(seed=1, extent=6.0)
    r.scene(obstacles=(3, 3))
    r.clear()
    assert ground.name in bpy.data.objects, 'clear() removed the ground'
    assert bot.root.name in bpy.data.objects, 'clear() removed the robot'
    assert r.created == []

    ## The ordering trap: scene() clears everything the randomizer made, lights
    ## included, so lights created first are destroyed by the next scene().
    r.lighting()
    assert [o for o in bpy.data.objects if o.type == 'LIGHT' and o.name.startswith('RND')]
    r.scene(obstacles=(2, 2))
    survivors = [o for o in bpy.data.objects
                 if o.type == 'LIGHT' and o.name.startswith('RND')]
    assert not survivors, 'lights should have been cleared -- document the order'
    ## the documented order works
    r.lighting()
    assert [o for o in bpy.data.objects if o.type == 'LIGHT' and o.name.startswith('RND')]
    r.clear()
    print('clear scoping OK')


def test_lineart_renders_and_restores():
    """
    Line art must leave the scene exactly as it found it.

    It overrides every material and whitens the world; leaking either into the
    next photorealistic frame would silently ruin the rest of the corpus, in a
    way that looks like a lighting bug rather than a leak.
    """
    fresh_scene()
    bot = Robot(arms=[], cameras='front')
    cam = bot.cameras['front']
    r = Randomizer(seed=5, extent=7.0)
    obs = r.scene(obstacles=(3, 5))
    r.materials(obs)
    r.lighting()
    r.world()
    r.camera(cam, looking_at=(0, 0, 0.6), distance=(6, 8))
    bpy.context.view_layer.update()

    scene = bpy.context.scene
    before = {
        'view_transform': scene.view_settings.view_transform,
        'override': bpy.context.view_layer.material_override,
        'freestyle': scene.render.use_freestyle,
        'world': scene.world,
    }
    line = LineArt(thickness=1.2)
    path = line.render(cam, os.path.join('/tmp', 'lineart-test.png'),
                       resolution=(W, H))
    assert os.path.isfile(path), path

    assert scene.view_settings.view_transform == before['view_transform'], \
        'view transform leaked'
    assert bpy.context.view_layer.material_override is before['override'], \
        'material override leaked'
    assert scene.render.use_freestyle == before['freestyle'], 'freestyle leaked'
    assert scene.world is before['world'], 'world leaked'

    ## the drawing is mostly white with a little ink, not a grey photograph
    from PIL import Image
    img = Image.open(path).convert('L')
    px = list(img.getdata())
    white = sum(1 for v in px if v > 240) / len(px)
    ink = sum(1 for v in px if v < 100) / len(px)
    print('line art: %.1f%% white, %.1f%% ink' % (white * 100, ink * 100))
    assert white > 0.5, 'background should be white, got %.2f' % white
    assert ink > 0.0005, 'no strokes were drawn'
    print('line art OK')


def test_dataset_write_and_verify():
    """A sample is files plus the manifest entry that explains them."""
    ds = Dataset(OUT, passes=('rgb', 'depth'), overwrite=True)
    ds.label('obstacle', 7)
    rgb = '/tmp/ds-rgb.png'
    depth = '/tmp/ds-depth.png'
    from PIL import Image
    Image.new('RGB', (W, H), (30, 60, 90)).save(rgb)
    Image.new('RGB', (W, H), (10, 10, 10)).save(depth)
    entry = ds.write(0, seed=123, meta={'note': 'x'}, rgb=rgb, depth=depth)
    ds.close()

    assert entry['seed'] == 123
    assert entry['labels']['obstacle'] == 7
    assert not os.path.isfile(rgb), 'source should be moved, not copied'
    assert len(ds.entries()) == 1
    info = json.load(open(os.path.join(OUT, 'dataset.json')))
    assert info['samples'] == 1 and info['passes'] == ['rgb', 'depth']
    ## labels travel with every sample, so a corpus relabelled midway stays readable
    assert 'labels' in entry
    print('write and manifest OK')


def test_dataset_refuses_incomplete():
    """A partial sample is worse than a missing one: it biases the set."""
    ds = Dataset(OUT + '-partial', passes=('rgb', 'depth'), overwrite=True)
    from PIL import Image
    rgb = '/tmp/ds-only.png'
    Image.new('RGB', (W, H), (30, 60, 90)).save(rgb)
    try:
        ds.write(0, rgb=rgb)
        raise AssertionError('should have refused a sample missing a pass')
    except ValueError as e:
        print('refused as expected: %s' % e)
    try:
        ds.write(1, rgb='/tmp/does-not-exist.png', depth=rgb)
        raise AssertionError('should have refused a missing file')
    except FileNotFoundError:
        pass
    ds.close()
    print('incomplete sample OK')


def test_verify_catches_misalignment_and_darkness():
    """
    Verification looks at the pixels, because the failures that matter do not
    show up in the file listing.
    """
    from PIL import Image
    ds = Dataset(OUT + '-bad', passes=('rgb', 'depth'), overwrite=True)
    ## sample 0: modalities disagree on size
    a, b = '/tmp/bad-a.png', '/tmp/bad-b.png'
    Image.new('RGB', (W, H), (120, 120, 120)).save(a)
    Image.new('RGB', (W // 2, H), (10, 10, 10)).save(b)
    ds.write(0, rgb=a, depth=b)
    ## sample 1: aligned, but the render was never lit
    c, d = '/tmp/bad-c.png', '/tmp/bad-d.png'
    Image.new('RGB', (W, H), (0, 0, 0)).save(c)
    Image.new('RGB', (W, H), (10, 10, 10)).save(d)
    ds.write(1, rgb=c, depth=d)
    ds.close()

    problems = ds.verify()
    print('verify found:', problems)
    assert any('disagree on size' in p for p in problems), problems
    assert any('tonal range' in p for p in problems), problems

    ## a good sample passes both checks
    ok = Dataset(OUT + '-good', passes=('rgb',), overwrite=True)
    import random
    good = '/tmp/good.png'
    rng = random.Random(0)
    img = Image.new('RGB', (W, H))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                 for _ in range(W * H)])
    img.save(good)
    ok.write(0, rgb=good)
    ok.close()
    assert ok.verify() == [], ok.verify()
    print('verification OK')


def test_shard_manifests_merge():
    """Shards write separate manifests; merging orders them by index."""
    from dataset import merge_manifests, read_manifest
    root = OUT + '-shards'
    os.system('rm -rf %s' % root)
    from PIL import Image
    made = []
    for shard in range(3):
        ds = Dataset(root, passes=('rgb',), overwrite=False,
                     manifest='manifest.%03d.jsonl' % shard)
        for i in range(shard, 9, 3):          # disjoint indices, as the launcher splits them
            path = '/tmp/sh-%d.png' % i
            Image.new('RGB', (8, 8), (i * 20 % 256, 40, 60)).save(path)
            ds.write(i, rgb=path)
            made.append(i)
        ds.close()

    merged = merge_manifests(root)
    assert merged == 9, merged
    entries = read_manifest(os.path.join(root, 'manifest.jsonl'))
    ## sorted by index, not by shard: the deterministic split by position must
    ## mean the same thing however many workers produced the corpus
    assert [e['index'] for e in entries] == list(range(9)), [e['index'] for e in entries]
    ## shard manifests are consumed
    import glob
    assert not glob.glob(os.path.join(root, 'manifest.0*.jsonl'))
    print('shard merge OK')


def test_merge_without_shards_is_harmless():
    """
    Merging a corpus that was never sharded must not truncate it.

    This is a real bug that shipped: with one worker the generator writes the
    real manifest directly, the merge step found no shard files, and wrote an
    empty one over the top -- leaving a corpus whose files all exist and whose
    index is empty.
    """
    from dataset import merge_manifests, read_manifest
    root = OUT + '-noshards'
    os.system('rm -rf %s' % root)
    from PIL import Image
    ds = Dataset(root, passes=('rgb',), overwrite=True)
    for i in range(4):
        path = '/tmp/ns-%d.png' % i
        Image.new('RGB', (8, 8), (10 * i, 20, 30)).save(path)
        ds.write(i, rgb=path)
    ds.close()

    kept = merge_manifests(root)
    assert kept == 4, kept
    assert len(read_manifest(os.path.join(root, 'manifest.jsonl'))) == 4
    print('merge without shards OK')


def test_prune_drops_only_the_bad():
    """One dead sample must not invalidate the corpus."""
    from dataset import indices_in
    from PIL import Image
    root = OUT + '-prune'
    os.system('rm -rf %s' % root)
    ds = Dataset(root, passes=('rgb',), overwrite=True)
    import random
    rng = random.Random(0)
    for i in range(5):
        path = '/tmp/pr-%d.png' % i
        img = Image.new('RGB', (16, 16))
        if i == 2:
            img.paste((128, 128, 128), (0, 0, 16, 16))      # featureless: unusable
        else:
            img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                         for _ in range(16 * 16)])
        img.save(path)
        ds.write(i, rgb=path)
    ds.close()

    problems = ds.verify()
    bad = indices_in(problems)
    print('verify flagged: %s' % sorted(bad))
    assert bad == {2}, (bad, problems)

    dropped = ds.prune(bad)
    assert dropped == 1
    kept = [e['index'] for e in ds.entries()]
    assert kept == [0, 1, 3, 4], kept
    assert ds.verify() == []
    ## files are left on disk, so a dropped sample can still be examined
    assert os.path.isfile(os.path.join(root, 'train', '000002.rgb.png'))
    print('prune OK')


def test_clear_frees_datablocks():
    """
    clear() must free the data, not just unlink the object.

    Removing an object orphans its mesh, and over a corpus of thousands of
    scenes those orphans accumulate into real memory -- and take the names with
    them, so identical scenes end up with differently-named objects.
    """
    r = Randomizer(seed=4, extent=6.0)
    ## Measured as a delta, not an absolute: earlier tests in this file leave
    ## their own lamps and meshes behind, and an absolute count would report
    ## their state rather than this one's.
    meshes_before = len(bpy.data.meshes)
    lights_before = len(bpy.data.lights)
    for _ in range(3):
        r.scene(obstacles=(4, 4))
        r.lighting(count=(2, 2))
        r.clear()
    mesh_leak = len(bpy.data.meshes) - meshes_before
    light_leak = len(bpy.data.lights) - lights_before
    print('after 3 build/clear cycles: %+d meshes, %+d lamps' % (mesh_leak, light_leak))
    assert mesh_leak <= 0, 'clear() leaked %d mesh datablocks' % mesh_leak
    assert light_leak <= 0, 'clear() leaked %d lamp datablocks' % light_leak
    print('datablock cleanup OK')


def test_end_to_end_sample():
    """One real sample, all four modalities, aligned."""
    fresh_scene()
    bot = Robot(arms=[], cameras='front', passes=('depth', 'segmentation'),
                out_dir='/tmp')
    cam = bot.cameras['front']
    r = Randomizer(seed=11, extent=8.0)
    obs = r.scene(obstacles=(4, 6), label=7)
    r.materials(obs)
    r.lighting(energy=(4.0, 14.0))
    r.world()
    r.camera(cam, looking_at=(0, 0, 0.6), distance=(6, 9))
    bpy.context.view_layer.update()

    ds = Dataset(OUT + '-e2e', passes=('rgb', 'depth', 'segmentation', 'lineart'),
                 overwrite=True)
    for token, index in SEGMENT_CLASSES.items():
        ds.label(token, index)
    ds.label('obstacle', 7)

    ## RGB on an engine that lights the scene; geometry passes on Cycles, which
    ## is the only engine exposing the object-index pass.
    set_render_engine('eevee')
    rgb = quick_render(cam, resolution_x=W, resolution_y=H,
                       output_path='/tmp/e2e.rgb.png')
    set_render_engine('cycles')
    configure_cycles(samples=4, bounces=0)
    cap = bot.capture(frame=0, resolution=(W, H))['front']
    line = LineArt().render(cam, '/tmp/e2e.lineart.png', resolution=(W, H))

    ds.write(0, cap, rgb=rgb, lineart=line, seed=11)
    ds.close()
    problems = ds.verify()
    print('end-to-end verify:', problems or 'clean')
    assert problems == [], problems

    entry = ds.entries()[0]
    assert sorted(entry['files']) == ['depth', 'lineart', 'rgb', 'segmentation']
    ## the semantic pass carries class indices, and the manifest says what they mean
    w, h, seg = read_pass(os.path.join(OUT + '-e2e', entry['files']['segmentation']))
    labels = sorted(set(int(round(v)) for v in seg))
    print('labels present: %s   manifest: %s' % (labels, sorted(entry['labels'].items())))
    assert all(v in entry['labels'].values() for v in labels), (labels, entry['labels'])
    print('end to end OK')


test_randomizer_is_deterministic()
test_randomizer_clears_only_its_own()
test_lineart_renders_and_restores()
test_dataset_write_and_verify()
test_dataset_refuses_incomplete()
test_verify_catches_misalignment_and_darkness()
test_shard_manifests_merge()
test_merge_without_shards_is_harmless()
test_prune_drops_only_the_bad()
test_clear_frees_datablocks()
test_end_to_end_sample()
print('dataset test OK')
