"""Concrete MetaWorld ML1 envs registered for offline meta-RL.

Goal-space ranges are the actual (glow, ghigh) ranges produced by MT1's
train tasks for each env (probed empirically against MetaWorld V3)."""
from rlkit.envs import register_env
from rlkit.envs.metaworld_base import (
    MetaWorldML1Base,
    _build_3d_grid_tasks,
    _build_2d_grid_tasks,
    _build_1d_tasks,
)


# Ranges below were probed from MT1.train_tasks for each MetaWorld V3 env.
# Reach: full 3D goal box.
# Push, Button-press: 2D goal (z fixed by env physics).
# Drawer-close: 1D goal (only x varies).


@register_env("metaworld-reach")
class MetaWorldReach(MetaWorldML1Base):
    METAWORLD_NAME = "reach-v3"
    GLOW = (-0.10, 0.80, 0.05)
    GHIGH = (0.10, 0.90, 0.30)

    @staticmethod
    def BUILD_TASKS():
        return _build_3d_grid_tasks(MetaWorldReach.GLOW, MetaWorldReach.GHIGH)


@register_env("metaworld-push")
class MetaWorldPush(MetaWorldML1Base):
    METAWORLD_NAME = "push-v3"
    GLOW = (-0.10, 0.80, 0.0194)
    GHIGH = (0.10, 0.90, 0.0194)

    @staticmethod
    def BUILD_TASKS():
        # z is degenerate → 2D grid in (x, y), fixed_dim = 2 (z)
        return _build_2d_grid_tasks(MetaWorldPush.GLOW, MetaWorldPush.GHIGH, fixed_dim=2)


@register_env("metaworld-button-press")
class MetaWorldButtonPress(MetaWorldML1Base):
    METAWORLD_NAME = "button-press-topdown-v3"
    GLOW = (-0.10, 0.80, 0.215)
    GHIGH = (0.10, 0.90, 0.215)

    @staticmethod
    def BUILD_TASKS():
        return _build_2d_grid_tasks(
            MetaWorldButtonPress.GLOW, MetaWorldButtonPress.GHIGH, fixed_dim=2
        )


@register_env("metaworld-drawer-close")
class MetaWorldDrawerClose(MetaWorldML1Base):
    METAWORLD_NAME = "drawer-close-v3"
    GLOW = (-0.10, 0.74, 0.09)
    GHIGH = (0.10, 0.74, 0.09)

    # MetaWorld V3 drawer-close has an ASYMMETRIC physical reachability: the drawer
    # slide / arm geometry caps x at roughly -0.085 on the negative side. Goals at
    # x ∈ {-0.0975, -0.0925, -0.0875} (default idx 30/31/32) are physically
    # unreachable — empirically confirmed by 6 SAC retries × diverse hyperparams
    # all converging to peak=0. We remap these 3 extreme-split tasks to reachable
    # positive-side intermediate x values, keeping the train/moderate split intact.
    _DRAWER_PATCH = {
        30: 0.0750,   # extreme positive, between moderate (+0.0725) and idx 35 (+0.0775) — converged ✓
        31: 0.0850,   # extreme positive, between idx 36 (+0.0825) and idx 37 (+0.0875) — converged ✓
        32: 0.0925,   # duplicate of idx 38 (+0.0925, proven rew=1860); SAC instability at +0.0900/+0.0950
                      # — user-accepted: replicate a known-good coord, collect with a different seed.
    }

    @staticmethod
    def BUILD_TASKS():
        # Only x varies → 1D grid, free_dim = 0
        pts = _build_1d_tasks(
            MetaWorldDrawerClose.GLOW, MetaWorldDrawerClose.GHIGH, free_dim=0
        )
        for idx, new_x in MetaWorldDrawerClose._DRAWER_PATCH.items():
            pts[idx] = pts[idx].copy()
            pts[idx][0] = float(new_x)
        return pts
