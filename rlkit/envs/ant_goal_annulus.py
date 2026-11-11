"""Annulus (inverted-split) variant of Ant-Goal for NeurIPS rebuttal experiment X7.

Reuses the exact 40 goals and the exact offline dataset of 'ant-goal-custom'
(./offline_dataset/ant-goal-custom/goal_idx{0..39}); ONLY the
train/moderate/extreme partition is remapped so that the held-out (extreme)
tasks lie radially INSIDE the training envelope (the Section-K counterexample
on the real benchmark).

Absolute goal-index layout inherited from AntGoalEnvCustom.sample_tasks():
    idx  0- 9 : r=0.8, theta = linspace(-pi/2,  pi/2, 10)   (right half, inner)
    idx 10-19 : r=1.2, theta = linspace(-pi/2,  pi/2, 10)   (right half, mid)
    idx 20-29 : r=1.6, theta = linspace(-pi/2,  pi/2, 10)   (right half, outer)
    idx 30-39 : r=1.6, theta = linspace( pi/2, 3pi/2, 50)[:10]  (left, 90-123 deg)

Original split : train = 0-19 (r in {0.8, 1.2}), moderate = 20-29, extreme = 30-39.
Annulus split  : train = 10-29 (r in {1.2, 1.6} right half),
                 moderate = 30-39 (leftover left-half r=1.6),
                 extreme = 0-9 (inner r=0.8 arc)  <- OOD radially inside.
"""

from rlkit.envs.ant_goal import AntGoalEnvCustom
from rlkit.envs import register_env


@register_env('ant-goal-annulus')
class AntGoalEnvAnnulus(AntGoalEnvCustom):
    """Same goals, same dataset; only task_modes() is overridden."""

    def task_modes(self):
        return {
            'train': list(range(10, 30)),     # r=1.2 arc + r=1.6 right-half arc
            'moderate': list(range(30, 40)),  # r=1.6 left-half arc (leftover)
            'extreme': list(range(0, 10)),    # r=0.8 inner arc (OOD, inside envelope)
        }
