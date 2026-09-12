#!/usr/bin/env python3
"""
Tests for Stage 2: the expert, the observation encoding and the policy.

Runs under plain python3. `control.py` is free of `bpy` for the same reason
`drive.py` is: the control loop should be checkable without launching a
renderer, and a policy should be trainable somewhere with more cores than the
machine that made the corpus.

    ./control_test.py           # or: make test_control_policy
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import control as C                                           # noqa: E402

print('hello control test...')


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


class World:
    """
    A kinematic stand-in for the simulator.

    Integrates the twist exactly as the kinematic contact model does, which is
    enough to exercise the expert and the rollout harness. Yaw is measured from
    +Y, matching the simulator's convention -- getting that wrong gives a
    controller that steers 90 degrees off and still looks plausible in
    aggregate.
    """

    def __init__(self, pose=(0.0, 0.0, 0.0), dt=0.15):
        self.x, self.y, self.yaw = pose
        self.dt = dt

    def pose(self):
        return (self.x, self.y, 0.0, self.yaw)

    def step(self, v, omega):
        self.yaw += omega * self.dt
        self.x -= math.sin(self.yaw) * v * self.dt
        self.y += math.cos(self.yaw) * v * self.dt
        return self.pose()


# ---------------------------------------------------------------------------
# the expert
# ---------------------------------------------------------------------------

def test_expert_reaches_an_open_goal():
    world = World()
    expert = C.Expert()
    goal = (3.0, 4.0)
    result = C.rollout(world.step, lambda p: expert.act(p, goal), world.pose(),
                       goal, steps=80)
    assert result['reached'], result
    print('  reached in %d steps, final %.2fm' % (result['steps'],
                                                  result['final_distance']))


def test_expert_avoids_an_obstacle_in_the_way():
    """Repulsion has to actually deflect the path, not merely exist."""
    goal = (0.0, 6.0)
    blocker = (0.0, 3.0, 0.6)

    world = World()
    expert = C.Expert()
    result = C.rollout(world.step, lambda p: expert.act(p, goal, [blocker]),
                       world.pose(), goal, steps=140)
    clearance = min(math.hypot(p[0] - blocker[0], p[1] - blocker[1])
                    for p in result['path'])
    assert clearance > blocker[2], 'drove through the obstacle (%.2f)' % clearance
    assert result['reached'], result
    print('  cleared a blocking obstacle by %.2fm and still arrived'
          % (clearance - blocker[2]))


def test_expert_stops_on_arrival():
    expert = C.Expert()
    v, omega = expert.act((1.0, 1.0, 0.0, 0.0), (1.0, 1.1))
    assert close(v, 0.0) and close(omega, 0.0), (v, omega)
    print('  commands zero once inside the arrival radius')


def test_expert_turns_before_driving():
    """
    Speed falls off with heading error, and never reverses.

    A controller that drives at full speed while turning describes an arc it
    cannot tighten; one that reverses to reduce heading error oscillates.
    """
    expert = C.Expert()
    ## Goal directly behind: yaw 0 faces +Y, so a goal at -Y is 180 degrees off.
    v_behind, _ = expert.act((0.0, 0.0, 0.0, 0.0), (0.0, -5.0))
    v_ahead, _ = expert.act((0.0, 0.0, 0.0, 0.0), (0.0, 5.0))
    assert close(v_behind, 0.0), v_behind
    assert v_ahead > 0.9 * expert.max_v, v_ahead
    assert v_behind >= 0.0, 'expert reversed'
    print('  full speed ahead %.2f, stationary when facing away %.2f'
          % (v_ahead, v_behind))


def test_yaw_convention_matches_the_simulator():
    """
    Yaw is measured from +Y. A goal to the east requires a negative turn.

    This is the convention `drive.body_to_world` uses. An expert written against
    the +X convention still reaches goals in open space -- it spirals in -- so
    the bug hides until obstacles matter.
    """
    expert = C.Expert()
    _v, omega_east = expert.act((0.0, 0.0, 0.0, 0.0), (5.0, 0.0))
    _v, omega_west = expert.act((0.0, 0.0, 0.0, 0.0), (-5.0, 0.0))
    assert omega_east < 0, omega_east
    assert omega_west > 0, omega_west
    print('  east -> omega %.2f, west -> omega %+.2f' % (omega_east, omega_west))


# ---------------------------------------------------------------------------
# observations and actions
# ---------------------------------------------------------------------------

def test_observation_channels():
    """
    Ink is inverted to match Stage 1's convention; classes are split by index.

    Training on one encoding and driving with another fails in a way that looks
    like a control problem, which is why both paths call this one function.
    """
    lineart = np.ones((4, 5), dtype=np.float32)          # white paper
    lineart[1, 1] = 0.0                                  # one ink pixel
    seg = np.zeros((4, 5), dtype=np.float32)
    seg[2, 2] = C.OBSTACLE_CLASS
    seg[3, 4] = C.GOAL_CLASS

    obs = C.observation(lineart, seg)
    assert obs.shape == (3, 4, 5), obs.shape
    assert close(float(obs[0, 1, 1]), 1.0), 'ink should be 1 where the page is dark'
    assert close(float(obs[0, 0, 0]), 0.0), 'blank paper should be 0'
    assert close(float(obs[1, 2, 2]), 1.0) and close(float(obs[1, 3, 4]), 0.0)
    assert close(float(obs[2, 3, 4]), 1.0) and close(float(obs[2, 2, 2]), 0.0)
    ## No photograph anywhere in the input: that is the architecture's claim.
    assert obs.shape[0] == 3, obs.shape
    print('  3 channels: ink, obstacle, goal -- and no colour')


def test_action_round_trip():
    for action in ((0.0, 0.0), (C.MAX_V, C.MAX_OMEGA), (0.4, -0.7)):
        back = C.decode_action(C.encode_action(action))
        assert close(back[0], action[0], 1e-5), (action, back)
        assert close(back[1], action[1], 1e-5), (action, back)
    ## Out-of-range commands clamp rather than wrap.
    assert close(C.encode_action((99.0, 0.0))[0], 1.0)
    print('  actions round-trip and clamp')


# ---------------------------------------------------------------------------
# the policy
# ---------------------------------------------------------------------------

def test_control_gradients():
    """Analytic gradients against a numerical check, in double precision."""
    rng = np.random.default_rng(0)
    net = C.ControlNet(width=4, depth=3, seed=1, dtype=np.float64, hidden=6)
    x = rng.standard_normal((2, 3, 6, 8))
    y = rng.standard_normal((2, 2)) * 0.3

    pred = net.forward(x)
    net.backward(C.control_loss(pred, y)[1])
    analytic = [g.copy() for _p, g in net.params()]

    eps, worst = 1e-6, 0.0
    for k, (param, _g) in enumerate(net.params()):
        flat, grad = param.reshape(-1), analytic[k].reshape(-1)
        for i in range(0, flat.size, max(1, flat.size // 4)):
            old = flat[i]
            flat[i] = old + eps
            up = C.control_loss(net.forward(x), y)[0]
            flat[i] = old - eps
            down = C.control_loss(net.forward(x), y)[0]
            flat[i] = old
            numeric = (up - down) / (2 * eps)
            worst = max(worst, abs(numeric - grad[i])
                        / max(1e-9, abs(numeric) + abs(grad[i])))
    assert worst < 1e-6, worst
    print('  worst relative gradient error %.2e' % worst)


def test_commands_are_bounded_by_construction():
    """
    tanh, not a clip after the fact.

    An unbounded head can learn to emit a command the drive model silently
    saturates, and the loss never sees the difference.
    """
    net = C.ControlNet(width=4, depth=2, seed=3)
    huge = np.ones((4, 3, 6, 6), dtype=np.float32) * 50.0
    out = net.forward(huge)
    assert np.abs(out).max() <= 1.0 + 1e-6, np.abs(out).max()
    v, omega = net.act(huge[0])
    assert abs(v) <= C.MAX_V + 1e-6 and abs(omega) <= C.MAX_OMEGA + 1e-6
    print('  saturating input gives |v| <= %.2f, |omega| <= %.2f' % (abs(v), abs(omega)))


def test_policy_can_fit_a_tiny_set():
    """
    The network can at least memorise four frames.

    Not a result about control -- it is the sanity check that says a poor score
    on real data is about the data or the task, not a wiring fault that stops
    the model learning anything at all.
    """
    rng = np.random.default_rng(4)
    x = rng.random((4, 3, 8, 8)).astype(np.float32)
    y = np.array([[0.8, -0.5], [-0.3, 0.6], [0.1, 0.9], [-0.7, -0.2]],
                 dtype=np.float32)
    net = C.ControlNet(width=8, depth=3, seed=0, hidden=24)
    C.train_control(net, x, y, epochs=250, batch=4, lr=1e-2, log=None)
    error = float(np.abs(net.forward(x) - y).mean())
    assert error < 0.1, error
    print('  memorised 4 frames to MAE %.3f' % error)


def test_control_scores_expose_the_constant_predictor():
    """
    An expert that mostly drives forward makes the mean command a good guess.

    A policy that has learned nothing still posts a small mean error, so the
    constant predictor is reported beside every score.
    """
    targets = np.tile(np.array([[0.9, 0.0]], dtype=np.float32), (20, 1))
    targets[:, 1] = np.linspace(-0.5, 0.5, 20)
    constant = np.repeat(targets.mean(axis=0)[None, :], len(targets), axis=0)
    scored = C.control_scores(constant, targets)
    assert close(scored['skill'], 0.0, 1e-6), scored
    perfect = C.control_scores(targets, targets)
    assert close(perfect['skill'], 1.0, 1e-6), perfect
    print('  constant predictor scores skill %.2f, perfect %.2f'
          % (scored['skill'], perfect['skill']))


def test_rollout_reports_failure_honestly():
    """A policy that drives away must not be recorded as having arrived."""
    world = World()
    goal = (0.0, 5.0)
    away = C.rollout(world.step, lambda p: (C.MAX_V, 0.0), world.pose(), goal,
                     steps=4)
    assert not away['reached'], away
    assert away['closest'] <= 5.0, away
    ## closest is a floor on final: a policy that passes through the goal and
    ## keeps going should show the difference.
    world2 = World()
    through = C.rollout(world2.step, lambda p: (C.MAX_V, 0.0), (0.0, 0.0, 0.0, 0.0),
                        (0.0, 1.0), steps=60, arrive=0.01)
    assert through['closest'] < through['final_distance'], through
    print('  overshoot recorded: closest %.2f, final %.2f'
          % (through['closest'], through['final_distance']))


TESTS = [test_expert_reaches_an_open_goal, test_expert_avoids_an_obstacle_in_the_way,
         test_expert_stops_on_arrival, test_expert_turns_before_driving,
         test_yaw_convention_matches_the_simulator, test_observation_channels,
         test_action_round_trip, test_control_gradients,
         test_commands_are_bounded_by_construction, test_policy_can_fit_a_tiny_set,
         test_control_scores_expose_the_constant_predictor,
         test_rollout_reports_failure_honestly]


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
