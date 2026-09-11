#!../headless.py
print('hello fleet test...')
import bpy, os
import firmware

if not firmware.available():
    print('SKIP fleet test: %s' % firmware.why_unavailable())
    print('fleet test OK (skipped)')
    raise SystemExit(0)

ROOT = HERE
DRIVE = os.path.join(ROOT, 'boards', 'drive_link_node.c')
NAV = os.path.join(ROOT, 'boards', 'nav_node.c')

DT = 1.0 / 60.0
create_cube('GROUND', size=(8000, 8000, 0.2), location=(0, 0, -0.1))
bpy.context.view_layer.update()

LANE = [0]
def next_lane(step=200.0):
    LANE[0] += step
    return LANE[0]


def close(a, b, tol=1e-2):
    return abs(a - b) < tol


def two_board_robot(**kw):
    x = next_lane()
    bot = Robot(arms=[], cameras='none', max_accel=2.0)
    bot.root.location = (x, 0, 0.4)
    bot.enable_contact()
    bpy.context.view_layer.update()
    drive = bot.attach_firmware(DRIVE, name='drive-mcu', source_mode='body', **kw)
    nav = bot.attach_board(NAV, name='nav-mcu')
    return bot, drive, nav


def run(bot, ticks):
    for _ in range(ticks):
        bot.step(DT)


def test_loop_rate():
    """
    A grant is one loop period, not one robotsim tick.

    hostsim's timer_count() consumes the whole grant in a single call, so a
    firmware delay loop exits immediately however much time it was given and the
    board runs exactly one iteration per grant. Handing over a whole tick would
    run a 1 kHz control loop at 60 Hz while its own arithmetic still believed it
    was 1 kHz.
    """
    bot, drive, nav = two_board_robot()
    seconds = 20.0
    run(bot, int(seconds / DT))
    print('board grants=%d over %.0f s (1 kHz wants %.0f)'
          % (nav.grants, seconds, seconds * nav.loop_hz))
    assert abs(nav.grants - seconds * nav.loop_hz) < nav.loop_hz, nav.grants

    ## the observable consequence: the planner's 4-second legs actually fire
    legs = [l for l in nav.lines if 'leg ->' in l]
    print('legs fired:', len(legs))
    assert len(legs) == 5, legs
    ## and the clock still does not drift, to within one loop period
    assert abs(nav.elapsed - seconds) <= 1.0 / nav.loop_hz, nav.elapsed
    print('loop rate OK')


def test_boards_command_each_other():
    """The nav board decides, the drive board actuates, the robot moves."""
    bot, drive, nav = two_board_robot()
    run(bot, 240)                     # 4 s: first leg is 2000 counts
    print('after leg 1: y=%.3f' % bot.root.location.y)
    assert close(bot.root.location.y, 2.0, 0.1), bot.root.location.y
    assert '[drive] target=2000' in drive.board.console

    run(bot, 240)                     # second leg: 4000
    print('after leg 2: y=%.3f' % bot.root.location.y)
    assert bot.root.location.y > 3.0, bot.root.location.y
    assert '[drive] target=4000' in drive.board.console

    stats = drive.board.link_stats
    print('drive link stats:', stats)
    assert stats['received'] >= 2, stats
    assert stats['sent'] > 0, 'drive board never reported telemetry'
    assert bot.network.delivered > 0
    assert not bot.network.undelivered
    print('inter-board command OK')


def test_one_step_latency():
    """
    A message sent during a tick arrives at the start of the next one.

    Delivery happens only once every board has reached the same virtual time,
    which is what makes the result independent of the order boards are listed
    in. The cost is one tick of latency, which is about what a real link costs.
    """
    bot, drive, nav = two_board_robot()
    sent_at = received_at = None
    for tick in range(300):
        before_nav = len(nav.lines)
        before_drive = len(drive.board.lines)
        bot.step(DT)
        if sent_at is None and any('leg ->' in l for l in nav.lines[before_nav:]):
            sent_at = tick
        if received_at is None and any('target=' in l for l in drive.board.lines[before_drive:]):
            received_at = tick
        if sent_at is not None and received_at is not None:
            break
    print('nav sent on tick %s, drive acted on tick %s' % (sent_at, received_at))
    assert sent_at is not None and received_at is not None, (sent_at, received_at)
    assert received_at == sent_at + 1, (sent_at, received_at)
    print('one-step latency OK')


def test_router_and_undelivered():
    """Broadcast is the default; a router narrows it, and nothing vanishes."""
    bot, drive, nav = two_board_robot()
    ## point everything at the drive board: the drive board's own telemetry now
    ## has nowhere to go, and must be recorded rather than dropped silently
    bot.network.router = firmware.point_to_point('drive-mcu')
    run(bot, 300)
    stats = drive.board.link_stats
    print('routed: drive received=%d, undelivered=%d'
          % (stats['received'], len(bot.network.undelivered)))
    assert stats['received'] > 0, 'setpoints should still reach the drive board'
    assert bot.network.undelivered, 'the drive board`s own telemetry should be recorded'
    names = {name for name, _msg in bot.network.undelivered}
    assert 'drive-mcu' in names, names
    print('routing OK')


def test_link_faults():
    """A link that fails is a link the firmware has to cope with."""
    bot, drive, nav = two_board_robot()
    ## everything the drive board sends is lost: the planner goes deaf
    drive.board.fault_link_down(True)
    run(bot, 400)
    print('link down: nav lines ->', [l for l in nav.lines if 'silent' in l])
    assert any('link silent' in l for l in nav.lines), nav.lines
    ## the drive board still receives, so the robot still moves
    assert bot.root.location.y > 1.0, bot.root.location.y

    ## one message in four dropped, counted rather than hidden
    bot2, drive2, nav2 = two_board_robot()
    drive2.board.fault_link_drop_every(2)
    run(bot2, 400)
    stats = drive2.board.link_stats
    print('drop_every(2): sent=%d dropped=%d' % (stats['sent'], stats['dropped']))
    assert stats['dropped'] > 0, stats
    print('link faults OK')


def test_across_robots():
    """One bus, two robots: a supervisor on one commanding a base on another."""
    xa = next_lane()
    lead = Robot(arms=[], cameras='none', max_accel=2.0)
    lead.root.location = (xa, 0, 0.4)
    lead.enable_contact()
    bpy.context.view_layer.update()
    planner = lead.attach_board(NAV, name='planner')

    xb = next_lane()
    follower = Robot(arms=[], cameras='none', max_accel=2.0)
    follower.root.location = (xb, 0, 0.4)
    follower.enable_contact()
    bpy.context.view_layer.update()
    base = follower.attach_firmware(DRIVE, name='follower-drive', source_mode='body')

    net = firmware.connect(lead, follower, name='robot-to-robot')
    print(net)
    assert net.owner is None, 'a shared bus must not be owned by one robot'
    assert len(net.boards) == 2

    ## Neither robot delivers a bus it does not own: routing before the other
    ## robot has stepped would break the same-virtual-time invariant.
    for _ in range(480):
        lead.step(DT)
        follower.step(DT)
        net.deliver()

    print('follower moved to y=%.3f on the planner`s orders' % follower.root.location.y)
    assert '[drive] target=' in base.board.console, base.board.console[:200]
    assert follower.root.location.y > 1.0, follower.root.location.y
    assert net.delivered > 0
    print('cross-robot OK')


def test_shared_bus_is_not_self_delivered():
    """A robot owning no bus must not deliver one behind the caller's back."""
    xa = next_lane()
    a = Robot(arms=[], cameras='none')
    a.root.location = (xa, 0, 0.4)
    bpy.context.view_layer.update()
    a.attach_board(NAV, name='solo-nav')
    assert a.network.owner is a, 'a robot owns its own bus'

    xb = next_lane()
    b = Robot(arms=[], cameras='none')
    b.root.location = (xb, 0, 0.4)
    bpy.context.view_layer.update()
    b.attach_board(NAV, name='other-nav')

    net = firmware.connect(a, b)
    for _ in range(120):
        a.step(DT)
        b.step(DT)
    ## nobody has called deliver(), so nothing has moved
    assert net.delivered == 0, net.delivered
    moved = net.deliver()
    print('after an explicit deliver(): %d messages moved' % moved)
    assert moved > 0, 'messages should have been waiting'
    print('bus ownership OK')


test_loop_rate()
test_boards_command_each_other()
test_one_step_latency()
test_router_and_undelivered()
test_link_faults()
test_across_robots()
test_shared_bus_is_not_self_delivered()
print('fleet test OK')
