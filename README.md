# LACO: Latent-Conservative Optimization for Robust Offline Meta-Reinforcement Learning

Code for the paper of the same title. LACO adds two regularizers to the actor-critic
losses of a context-based offline meta-RL method and leaves the encoder unchanged:
smoothness penalties on the critic and the policy along the task latent, and a
conservative penalty that compares the critic's value under a task's own latent with its
values under the latents of the other tasks in the batch.

The method is in `rlkit/torch/agents/laco.py`. The baselines CSRO, UNICORN and ER-TRL are in
the same folder, and `laco_unicorn.py` applies LACO to the UNICORN backbone.

## Setup

Python 3.10 with MuJoCo 2.1 (`mujoco-py`), PyTorch, `gym==0.23.1`, `hydra-core==1.3.2`,
`gtimer`, `pandas`, `scipy`, and optionally `wandb`. Run all commands from the repository root.

## Data

Each task is trained with soft actor-critic, and its policy is rolled out at regular
checkpoints. `collect_data.py` does this for one task and writes to `./offline_dataset/`:

```bash
python collect_data.py env=ant-dir goal_idx=0
```

## Training

Add `seed=<n>` and, optionally, `use_wandb=true +util_params.wandb_project=<name>`.

| Environment | Command |
|---|---|
| Ant-Dir | `python launch_experiment.py algo=laco env=ant-dir` |
| Ant-Goal | `python launch_experiment.py algo=laco env=ant-goal algo_params.alpha_lc=4.2` |
| HalfCheetah-Vel | `python launch_experiment.py algo=laco env=cheetah-vel` |
| Humanoid-Dir | `python launch_experiment.py algo=laco env=humanoid-dir algo_params.lc_auto=true algo_params.lc_auto_lr=0.0001 algo_params.lc_target=0.9` |
| Walker-Friction | `python launch_experiment.py algo=laco env=walker-friction algo_params.lc_auto=true algo_params.lc_auto_lr=0.0001 algo_params.lc_target=0.9` |
| Hopper-Mass | `python launch_experiment.py algo=laco env=hopper-mass algo_params.lc_auto=true algo_params.lc_auto_lr=0.0001 algo_params.lc_target=0.5 algo_params.lc_auto_warmup=20000 algo_params.lc_auto_ema=0.999 algo_params.lc_auto_max_step=0.002` |

Baselines use `algo=csro`, `algo=unicorn` or `algo=ertrl`. Set `util_params.exp_name=<name>` to
name a run; outputs go to `./output/<env>/<exp_name>/seed<n>/`. For each run, the checkpoint
`agent.pth` holds the final weights.

## Evaluation

```bash
# return under contexts collected by other tasks' policies, and by the task's own policy
python scripts/eval_ood_behavior.py --env ant-dir --all-runs --algos laco,csro --seeds 0,1,2,3,4,5 \
    --ckpt-root ./output --out results_ant-dir.csv
# matched-policy return with ten contexts per task
python scripts/eval_iid_repeat.py --env ant-dir --algos laco,csro --ckpt-root ./output --out iid_ant-dir.csv
# critic along lines through latent space; latent displacement and critic sensitivity
python scripts/latent_value_path.py --env ant-dir --algos laco,csro --seeds 0,1,2,3,4,5 --ckpt-root ./output --out path_ant-dir.csv
python scripts/analyze_latent_value.py --env ant-dir --algo laco --cpu --ckpt-root ./output --out ./output/latent_value
```

Prefix a command with `python scripts/cpu_run.py` in place of `python` to evaluate on the CPU.
