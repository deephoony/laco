"""MetaWorld ML1 wrappers for offline meta-RL benchmarking.

Each env exposes 40 tasks split as:
    train     [0, 20)   — central goal positions   (ID)
    moderate  [20, 30)  — outer-shell goals near training set
    extreme   [30, 40)  — outer-shell goals far from training set

The split is built from a discretized goal-space grid:
    3D goal (e.g. reach):   5x5x5  cuboid grid + Chebyshev/Euclidean ranking
    2D goal (push, button): 7x7    grid in the two free dims
    1D goal (drawer-close): 40 evenly-spaced points along the free dim
"""
import numpy as np
from gym import Env
from gym.spaces import Box
from metaworld import MT1


# -------------------------------------------------------------------- task generators


def _build_3d_grid_tasks(glow, ghigh):
    """5x5x5 cuboid grid → 40 ordered goal points (train, moderate, extreme).

    Train (20):
        Inner 27 cuboids (Chebyshev ≤ 1), sorted by Euclidean dist to centre, 20 closest.
        = centre + 6 face + 12 edge + 1 vertex of inner 3x3x3.
    Moderate (10):
        Outer-shell cuboids (Chebyshev = 2), sorted ascending, 10 closest.
        = 6 face-of-outer + 4 edge cuboids.
    Extreme (10):
        Outer-shell cuboids, sorted descending, 10 farthest.
        = 8 vertex + 2 edge cuboids.
    """
    glow = np.asarray(glow, dtype=np.float64)
    ghigh = np.asarray(ghigh, dtype=np.float64)
    cell = (ghigh - glow) / 5.0
    cells = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                centre = glow + (np.array([i, j, k], dtype=np.float64) + 0.5) * cell
                cheb = max(abs(i - 2), abs(j - 2), abs(k - 2))
                eucl = float(np.sqrt((i - 2) ** 2 + (j - 2) ** 2 + (k - 2) ** 2))
                cells.append({"centre": centre, "cheb": cheb, "eucl": eucl, "idx": (i, j, k)})

    inner = sorted([c for c in cells if c["cheb"] <= 1],
                   key=lambda c: (c["eucl"], c["idx"]))[:20]
    outer = [c for c in cells if c["cheb"] == 2]
    moderate = sorted(outer, key=lambda c: (c["eucl"], c["idx"]))[:10]
    extreme = sorted(outer, key=lambda c: (-c["eucl"], c["idx"]))[:10]
    return [c["centre"] for c in inner + moderate + extreme]


def _build_2d_grid_tasks(glow, ghigh, fixed_dim):
    """7x7 grid in the two free dims (the third dim is fixed at the box centre).
    Inner 5x5 = 25 cuboids → 20 train (sorted by distance to centre).
    Outer ring 24 cuboids → 10 moderate (closest) + 10 extreme (farthest)."""
    glow = np.asarray(glow, dtype=np.float64)
    ghigh = np.asarray(ghigh, dtype=np.float64)
    free_dims = [d for d in range(3) if d != fixed_dim]
    free_low = glow[free_dims]
    free_high = ghigh[free_dims]
    cell = (free_high - free_low) / 7.0
    fixed_value = (glow[fixed_dim] + ghigh[fixed_dim]) / 2.0
    centre = (7 - 1) / 2.0  # = 3.0
    cells = []
    for i in range(7):
        for j in range(7):
            pos = np.zeros(3, dtype=np.float64)
            pos[fixed_dim] = fixed_value
            pos[free_dims[0]] = free_low[0] + (i + 0.5) * cell[0]
            pos[free_dims[1]] = free_low[1] + (j + 0.5) * cell[1]
            cheb = max(abs(i - centre), abs(j - centre))
            eucl = float(np.sqrt((i - centre) ** 2 + (j - centre) ** 2))
            cells.append({"centre": pos, "cheb": cheb, "eucl": eucl, "idx": (i, j)})

    inner = sorted([c for c in cells if c["cheb"] <= 2],
                   key=lambda c: (c["eucl"], c["idx"]))[:20]
    outer = [c for c in cells if c["cheb"] == 3]
    moderate = sorted(outer, key=lambda c: (c["eucl"], c["idx"]))[:10]
    extreme = sorted(outer, key=lambda c: (-c["eucl"], c["idx"]))[:10]
    return [c["centre"] for c in inner + moderate + extreme]


def _build_1d_tasks(glow, ghigh, free_dim):
    """40 evenly-spaced points along the free dim.

    MuJoCo-style index split:
        train    = indices 10..29  (central band)
        moderate = indices 5..9, 30..34
        extreme  = indices 0..4, 35..39
    """
    glow = np.asarray(glow, dtype=np.float64)
    ghigh = np.asarray(ghigh, dtype=np.float64)
    fixed_dims = [d for d in range(3) if d != free_dim]
    cell = (ghigh[free_dim] - glow[free_dim]) / 40.0
    pts = []
    for i in range(40):
        pos = np.zeros(3, dtype=np.float64)
        pos[free_dim] = glow[free_dim] + (i + 0.5) * cell
        for d in fixed_dims:
            pos[d] = (glow[d] + ghigh[d]) / 2.0
        pts.append(pos)
    train = list(range(10, 30))
    moderate = list(range(5, 10)) + list(range(30, 35))
    extreme = list(range(0, 5)) + list(range(35, 40))
    return [pts[i] for i in (train + moderate + extreme)]


# -------------------------------------------------------------------- env base


class MetaWorldML1Base(Env):
    """Common wrapper for MetaWorld ML1 tasks under our train/moderate/extreme split.

    Subclasses set:
        METAWORLD_NAME : str (e.g. "reach-v3")
        BUILD_TASKS    : callable returning list of 40 (3,)-arrays of goal positions
    """
    METAWORLD_NAME = None
    BUILD_TASKS = None  # subclass: () -> List[np.ndarray]
    _GOAL_DIMS = 3      # last 3 obs dims are the goal in MetaWorld V3

    def __init__(self, idx=0, n_tasks=40, max_episode_steps=200, **kwargs):
        assert n_tasks == 40, "Only n_tasks=40 supported"
        assert self.METAWORLD_NAME is not None, "subclass must set METAWORLD_NAME"

        mt1 = MT1(self.METAWORLD_NAME)
        env_cls = mt1.train_classes[self.METAWORLD_NAME]
        self._inner_env = env_cls()
        self._template_task = mt1.train_tasks[0]
        self._inner_env.set_task(self._template_task)
        self._inner_env.reset()

        # Goal points and current task
        self._goal_points = type(self).BUILD_TASKS()
        assert len(self._goal_points) == 40
        self.tasks = [{"goal": np.asarray(g, dtype=np.float64)} for g in self._goal_points]
        self._goal_idx = idx
        self._task = self.tasks[idx]
        self._goal = self._task["goal"]

        self.num_tasks = 40
        self._max_episode_steps = max_episode_steps
        self._step = 0

        # Observation space without the trailing 3 goal dims
        full_low = self._inner_env.observation_space.low[:-self._GOAL_DIMS]
        full_high = self._inner_env.observation_space.high[:-self._GOAL_DIMS]
        self.observation_space = Box(full_low.astype(np.float32),
                                     full_high.astype(np.float32),
                                     dtype=np.float32)
        self.action_space = self._inner_env.action_space

    # ---- task management (matches existing rlkit envs) ----

    def reset_task(self, idx):
        self._goal_idx = idx
        self._task = self.tasks[idx]
        self._goal = self._task["goal"]
        self.reset()

    def get_all_task_idx(self):
        return list(range(self.num_tasks))

    def get_idx(self):
        return self._goal_idx

    def get_task(self):
        return self._task

    def task_modes(self):
        return {
            "train":    list(range(0, 20)),
            "moderate": list(range(20, 30)),
            "extreme":  list(range(30, 40)),
        }

    def get_mode(self):
        for name, idxs in self.task_modes().items():
            if self._goal_idx in idxs:
                return name
        return "unknown"

    # ---- gym interface ----

    def _strip_goal(self, obs):
        return obs[:-self._GOAL_DIMS].astype(np.float32)

    def reset(self, **_):
        self._step = 0
        # Re-attach the template task so the simulation state is well-formed,
        # then override the target to our cuboid centre. Reward computation
        # in MetaWorld V3 uses self._target_pos directly, so this is safe.
        self._inner_env.set_task(self._template_task)
        result = self._inner_env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        self._inner_env._target_pos = np.asarray(self._goal, dtype=np.float64)
        return self._strip_goal(obs)

    def step(self, action):
        result = self._inner_env.step(action)
        if len(result) == 5:
            obs, reward, term, trunc, info = result
        else:
            obs, reward, term, info = result
            trunc = False
        self._step += 1
        done = bool(term) or self._step >= self._max_episode_steps
        # Provide minimal "success" + raw info passthrough; the SAC trainer
        # reads info["success"] directly for wandb logging.
        out_info = {
            "success": float(info.get("success", 0.0)),
            "near_object": float(info.get("near_object", 0.0)),
            "grasp_success": float(info.get("grasp_success", 0.0)),
            "obj_to_target": float(info.get("obj_to_target", 0.0)),
            "goal_idx": int(self._goal_idx),
            "goal_xyz": np.asarray(self._goal, dtype=np.float64).tolist(),
        }
        # numpy scalar (0-d) for reward so rlkit's `r.shape` check (rollout context update) works.
        return self._strip_goal(obs), np.float32(reward), bool(done), out_info

    def seed(self, seed=None):
        if hasattr(self._inner_env, "seed"):
            self._inner_env.seed(seed)
        self._np_seed = seed
        return [seed]

    def render(self, *args, **kwargs):
        return self._inner_env.render(*args, **kwargs)
