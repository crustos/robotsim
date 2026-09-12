"""
Stage 2: turning the abstraction into a command.

Stage 1 converts a photograph into a line drawing plus a semantic map. This is
the other half of the chain: line drawing plus semantic map in, a drive command
out. It never sees a photograph, so there is no texture for it to overfit to --
that is the whole argument of the architecture, and this module is where it
either holds or does not.

    expert = Expert()
    action = expert.act(pose, goal, obstacles)      # privileged
    policy = ControlNet()                           # sees only the abstraction
    train_control(policy, observations, actions)

WHERE THE LABELS COME FROM
--------------------------
Behaviour cloning from a privileged expert. The expert is given the true pose
of the robot, the true position of the goal and the true position of every
obstacle, and computes a command from geometry. The student is given the
rendered abstraction from the robot's own camera and is trained to produce the
same command. The expert is not a good driver in any general sense -- it is a
potential field, and potential fields get stuck -- but it is deterministic,
inspectable, and it never needs to be right for the comparison to be meaningful.
What is being measured is whether the abstraction carries enough information to
recover the expert's decision, not whether the expert's decision was wise.

The honest failure mode to watch for: a cloned policy can score well on held-out
*frames* and still fail the moment it drives, because its own errors take it to
states the expert never visited and the training set therefore never contained.
Frame-level regression error is reported here, but `rollout` is the number that
matters, and the two are reported together for exactly that reason.

WHAT THE POLICY SEES
--------------------
Three channels, all derivable from Stage 1's output and none from a photograph:

    0  ink          -- the line drawing
    1  obstacle     -- semantic map, obstacle class
    2  goal         -- semantic map, goal class

The goal has to be visible for the task to be solvable at all. A policy asked to
drive to a target it cannot see is being asked to guess, and it will learn the
average direction of the goal over the training set -- which looks like progress
on frame error and is worthless in a rollout.
"""

import math

import numpy as np

#: Semantic indices the control scenes use. Distinct from the perception
#: corpus's classes because a control frame is a different scene, and reusing an
#: index across corpora is how a policy ends up steering towards a mug.
OBSTACLE_CLASS = 7
GOAL_CLASS = 9

#: Commands are normalised into [-1, 1] before the network sees them. A network
#: regressing raw metres per second and radians per second is fitting two
#: quantities whose scales differ by an order of magnitude, and the loss is then
#: dominated by whichever happens to be larger.
MAX_V = 1.2
MAX_OMEGA = 1.5


# ---------------------------------------------------------------------------
# the expert
# ---------------------------------------------------------------------------

class Expert:
    """
    A potential-field controller with privileged access to the scene.

    Attraction to the goal, repulsion from obstacles, and a speed that falls off
    with heading error so the robot turns before it drives. Deliberately simple:
    every term is one line and can be checked by hand, which matters because a
    cloned policy can only be as good as the thing it is cloning, and a bug in
    the expert shows up as an unexplained ceiling on the student.
    """

    def __init__(self, max_v=MAX_V, max_omega=MAX_OMEGA, influence=1.6,
                 repulsion=1.4, turn_gain=2.0, arrive=0.45):
        self.max_v = max_v
        self.max_omega = max_omega
        self.influence = influence
        self.repulsion = repulsion
        self.turn_gain = turn_gain
        self.arrive = arrive

    def act(self, pose, goal, obstacles=()):
        """
        (v, omega) for one step. `pose` is (x, y, yaw); obstacles are (x, y, r).
        """
        x, y, yaw = pose[0], pose[1], pose[3] if len(pose) > 3 else pose[2]
        to_goal = np.array([goal[0] - x, goal[1] - y], dtype=np.float64)
        distance = float(np.linalg.norm(to_goal))
        if distance < self.arrive:
            return (0.0, 0.0)

        direction = to_goal / max(distance, 1e-9)

        ## Repulsion falls off to zero at `influence` rather than with 1/r, so a
        ## distant obstacle has no effect at all. A field that never quite
        ## reaches zero makes every command depend on every object in the scene,
        ## which is not something the abstraction can convey.
        for ox, oy, radius in obstacles:
            away = np.array([x - ox, y - oy], dtype=np.float64)
            gap = float(np.linalg.norm(away)) - radius
            if gap < self.influence:
                strength = self.repulsion * (1.0 - max(gap, 0.0) / self.influence)
                direction = direction + (away / max(np.linalg.norm(away), 1e-9)) * strength

        direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
        desired = math.atan2(direction[1], direction[0])
        ## Yaw is measured from +Y in this simulator, not +X.
        error = wrap(desired - (yaw + math.pi / 2))

        omega = clamp(self.turn_gain * error, -self.max_omega, self.max_omega)
        ## Drive only in the direction being faced: cos of the heading error,
        ## floored at zero so the robot never reverses to reduce it.
        v = self.max_v * max(0.0, math.cos(error))
        return (v, omega)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, low, high):
    return max(low, min(high, value))


def encode_action(action):
    """(v, omega) -> [-1, 1]^2."""
    v, omega = action
    return np.array([clamp(v / MAX_V, -1, 1), clamp(omega / MAX_OMEGA, -1, 1)],
                    dtype=np.float32)


def decode_action(vector):
    return (float(vector[0]) * MAX_V, float(vector[1]) * MAX_OMEGA)


# ---------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------

def observation(lineart, segmentation, obstacle=OBSTACLE_CLASS, goal=GOAL_CLASS):
    """
    Stack a line drawing and a semantic map into the policy's input.

    Takes arrays rather than paths so the same function serves training (from a
    corpus) and closed-loop control (from a live render), which is what keeps
    the two from drifting apart -- a policy trained on one encoding and driven
    with another fails in a way that looks like a control problem.

    `lineart` is greyscale in [0,1] with white paper; it is inverted here so 1
    means ink, matching Stage 1's output convention.
    """
    ink = 1.0 - np.asarray(lineart, dtype=np.float32)
    seg = np.asarray(segmentation, dtype=np.float32).round()
    return np.stack([ink,
                     (seg == obstacle).astype(np.float32),
                     (seg == goal).astype(np.float32)])


# ---------------------------------------------------------------------------
# the policy
# ---------------------------------------------------------------------------

class ControlNet:
    """
    Abstraction in, normalised command out.

    The spatial map has to collapse before the output layer, because a drive
    command is one decision about the whole frame rather than a per-pixel one.
    *How* it collapses turns out to be the whole ballgame: a global mean gives a
    policy that learns throttle and cannot steer, because the mean of a channel
    says how much goal is visible and not which side it is on. Pooling into
    horizontal bands keeps that, and steering is entirely a question about which
    side the goal is on.

    Pooling after the convolutions rather than striding through them keeps the
    resolution where the thin strokes are -- the same reasoning `LineArtNet`
    uses, for the same reason.
    """

    def __init__(self, width=12, depth=3, seed=0, in_ch=3, dtype=np.float32,
                 hidden=32, bins=8):
        from perception import Conv2d, ReLU, BandPool, Linear
        rng = np.random.default_rng(seed)
        self.dtype = dtype
        self.trunk = []
        channels = in_ch
        for _ in range(max(1, depth - 1)):
            self.trunk.append(Conv2d(channels, width, rng=rng, dtype=dtype))
            self.trunk.append(ReLU())
            channels = width
        ## Banded rather than global: see BandPool. A single mean over the
        ## frame cannot say which side the goal is on, and steering is entirely
        ## a question about which side the goal is on. Eight bins rather than
        ## four because the difference is measurable -- val MAE 0.111 against
        ## 0.131 on the same split -- and because the quantity being recovered
        ## is a horizontal position, so the resolution of the pooling is the
        ## resolution of the answer.
        self.pool = BandPool(bins=bins)
        self.fc1 = Linear(channels * bins, hidden, rng=rng, dtype=dtype)
        self.act1 = ReLU()
        self.fc2 = Linear(hidden, 2, rng=rng, dtype=dtype)

    def params(self):
        out = []
        for layer in self.trunk:
            out.extend(layer.params())
        out.extend(self.fc1.params())
        out.extend(self.fc2.params())
        return out

    def forward(self, x):
        for layer in self.trunk:
            x = layer.forward(x)
        h = self.act1.forward(self.fc1.forward(self.pool.forward(x)))
        ## tanh, so the command is bounded by construction rather than by a
        ## clip after the fact. An unbounded head can learn to emit a command
        ## the drive model will silently saturate, and the loss never sees it.
        self.raw = self.fc2.forward(h)
        return np.tanh(self.raw)

    def backward(self, grad):
        grad = grad * (1.0 - np.tanh(self.raw) ** 2)
        grad = self.fc1.backward(self.act1.backward(self.fc2.backward(grad)))
        grad = self.pool.backward(grad)
        for layer in reversed(self.trunk):
            grad = layer.backward(grad)
        return grad

    def act(self, obs):
        """One observation (C,H,W) -> (v, omega) in real units."""
        out = self.forward(np.asarray(obs, dtype=self.dtype)[None])
        return decode_action(out[0])

    def save(self, path):
        np.savez(path, **{'p%d' % i: p for i, (p, _g) in enumerate(self.params())})
        return path

    def load(self, path):
        data = np.load(path)
        for i, (p, _g) in enumerate(self.params()):
            p[...] = data['p%d' % i]
        return self


def control_loss(pred, targets):
    """Mean squared error over the two normalised command channels."""
    diff = pred - targets
    loss = float((diff ** 2).mean())
    return loss, (2.0 * diff / diff.size).astype(pred.dtype)


def control_scores(pred, targets):
    """
    Per-channel error, and the score of predicting the training mean.

    The constant baseline matters more here than anywhere else in this codebase.
    An expert that mostly drives forward produces a command set whose mean is
    close to most of its members, so a policy that has learned nothing at all
    reports a small mean error. Reported side by side, always.
    """
    err = np.abs(pred - targets)
    constant = np.repeat(targets.mean(axis=0)[None, :], len(targets), axis=0)
    base = np.abs(constant - targets)
    return {'v_mae': float(err[:, 0].mean()),
            'omega_mae': float(err[:, 1].mean()),
            'mae': float(err.mean()),
            'baseline_mae': float(base.mean()),
            ## Fraction of the constant-predictor's error that the model
            ## removes. Negative means worse than predicting the mean.
            'skill': float(1.0 - err.mean() / max(1e-9, base.mean()))}


def train_control(net, x, y, epochs=20, batch=8, lr=3e-3, seed=0, log=print,
                  val=None):
    from perception import Adam
    opt = Adam(net.params(), lr=lr)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        total, batches = 0.0, 0
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            pred = net.forward(x[idx])
            loss, grad = control_loss(pred, y[idx])
            net.backward(grad)
            opt.step()
            total += loss
            batches += 1
        entry = {'epoch': epoch, 'loss': total / max(1, batches)}
        entry.update({'train_' + k: v for k, v in
                      control_scores(net.forward(x), y).items()})
        if val is not None:
            entry.update({'val_' + k: v for k, v in
                          control_scores(net.forward(val[0]), val[1]).items()})
        history.append(entry)
        if log:
            message = ('epoch %2d  loss %.4f  train MAE %.3f (mean %.3f)'
                       % (entry['epoch'], entry['loss'], entry['train_mae'],
                          entry['train_baseline_mae']))
            if val is not None:
                message += ('  |  val MAE %.3f (mean %.3f, skill %+.2f)'
                            % (entry['val_mae'], entry['val_baseline_mae'],
                               entry['val_skill']))
            log(message)
    return history


# ---------------------------------------------------------------------------
# closed loop
# ---------------------------------------------------------------------------

def rollout(step_fn, policy_fn, start, goal, steps=40, arrive=0.45):
    """
    Drive a policy and report whether it got there.

    `step_fn(v, omega) -> pose` advances the world one tick and returns the new
    pose; `policy_fn(pose) -> (v, omega)` is the thing under test. Kept abstract
    so the same harness scores the expert, the cloned policy driven from ground
    truth abstraction, and the cloned policy driven from Stage 1's prediction --
    which is the comparison that says what the perception stage costs.
    """
    pose = start
    path = [pose]
    best = float('inf')
    for _ in range(steps):
        v, omega = policy_fn(pose)
        pose = step_fn(v, omega)
        path.append(pose)
        distance = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
        best = min(best, distance)
        if distance < arrive:
            return {'reached': True, 'steps': len(path) - 1,
                    'final_distance': distance, 'closest': best, 'path': path}
    final = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
    return {'reached': False, 'steps': steps, 'final_distance': final,
            'closest': best, 'path': path}
