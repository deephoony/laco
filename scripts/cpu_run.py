"""Run an evaluation script with the networks on CPU instead of a GPU.

Rollout evaluation is batch-1 inference through small MLPs plus MuJoCo stepping,
so it does not need a GPU; on a GPU that is shared with training jobs, dozens of
evaluation processes only add time-slicing contention that slows the training
runs down. This wrapper replaces eval_ood_behavior.build_algorithm with a
use_gpu=false version before the target script imports it.

  python scripts/cpu_run.py scripts/eval_ood_behavior.py --env ... [args]
  python scripts/cpu_run.py scripts/eval_iid_repeat.py --env ...
  python scripts/cpu_run.py scripts/latent_value_path.py --env ...
(--gpu-id is accepted and ignored.)
"""
import os
import runpy
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)

import eval_ood_behavior as E  # noqa: E402


def build_cpu(env_group, algo, gpu_id, extra_overrides=()):
    from hydra import compose, initialize_config_dir
    import launch_experiment
    with initialize_config_dir(config_dir=os.path.join(REPO, "cfgs"),
                               version_base="1.3", job_name="eval_cpu"):
        cfg = compose(config_name="experiment",
                      overrides=[f"env={env_group}", f"algo={algo}",
                                 "use_wandb=false", "use_gpu=false", "gpu_id=0",
                                 *extra_overrides])
    return launch_experiment.initialize(cfg), cfg


E.build_algorithm = build_cpu

target = sys.argv[1]
sys.argv = sys.argv[1:]
if os.path.basename(target) == "eval_ood_behavior.py":
    E.main()
else:
    runpy.run_path(target, run_name="__main__")
