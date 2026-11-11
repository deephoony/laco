"""Few-shot generalization to out-of-distribution behavior policies (UNICORN Sec. 3.3).

For each test task M^j the context is collected by a SAC behavior policy pi^i that
was trained on a *different* task M^i, executed inside M^j. The task identity is
therefore still recoverable from the rewards -- only the behavior that produced
the transitions is off-distribution. The meta-policy is then scored by a
deterministic rollout in M^j conditioned on the posterior inferred from that
context. i == j reproduces the in-distribution reference under identical
machinery.

The datasets keep one SAC checkpoint per task (`goal_idx<i>/actor_step<T>.pkl`,
the converged policy), so the behavior-policy pool spans tasks at a fixed
expert quality; UNICORN additionally varies the training iteration t, which our
data does not retain.

Usage:
  python scripts/eval_ood_behavior.py --env ant-dir --algo laco --seed 0 \
      --ckpt-root ./output/checkpoints --out results.csv
  python scripts/eval_ood_behavior.py --env ant-dir --all-runs --n-policies 10
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

# Rollouts are batch-1 forward passes through ~0.1M-parameter nets, so intra-op
# threading buys nothing and costs a lot: eight of these processes at torch's
# default of 64 threads each drove levine's load average to 382 on 128 cores and
# made per-checkpoint time swing between 2 and 35 minutes.
torch.set_num_threads(1)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hydra import compose, initialize_config_dir  # noqa: E402
import rlkit.torch.pytorch_util as ptu  # noqa: E402
from rlkit.torch.pytorch_sac.agent.actor import DiagGaussianActor  # noqa: E402

# exp_name as it appears in the checkpoint tree, per hydra algo group
ALGO_DIR = {"laco": "laco", "csro": "CSRO", "unicorn": "unicorn", "ertrl": "ertrl"}
# env group name -> env_name used for the dataset directory
ENV_GROUPS = ["ant-dir", "ant-goal", "cheetah-vel", "humanoid-dir",
              "hopper-mass", "hopper-friction", "walker-mass", "walker-friction"]
# hydra env group -> cfg.env_name (the name used for dataset and output dirs)
ENV_NAME = {g: (g + "-custom" if g in ("ant-dir", "ant-goal", "cheetah-vel",
                                       "humanoid-dir") else g)
            for g in ENV_GROUPS}

SAC_HIDDEN_DIM = 1024
SAC_HIDDEN_DEPTH = 2
SAC_LOG_STD_BOUNDS = [-5, 2]


def build_algorithm(env_group, algo, gpu_id, extra_overrides=()):
    """Reconstruct the training-time algorithm object (nets on the right device)."""
    import launch_experiment
    with initialize_config_dir(config_dir=os.path.join(REPO, "cfgs"),
                               version_base="1.3", job_name="eval_ood"):
        cfg = compose(config_name="experiment",
                      overrides=[f"env={env_group}", f"algo={algo}",
                                 "use_wandb=false", "use_gpu=true",
                                 f"gpu_id={gpu_id}", *extra_overrides])
    return launch_experiment.initialize(cfg), cfg


def load_behavior_actor(data_dir, task_idx, obs_dim, action_dim, cache):
    """SAC actor that generated task `task_idx`'s data (converged checkpoint)."""
    if task_idx in cache:
        return cache[task_idx]
    d = os.path.join(data_dir, f"goal_idx{task_idx}")
    ckpts = sorted(f for f in os.listdir(d) if f.startswith("actor_step"))
    if not ckpts:
        raise FileNotFoundError(f"no actor checkpoint in {d}")
    # highest step = converged policy
    path = os.path.join(d, max(ckpts, key=lambda f: int(f[len("actor_step"):-4])))
    actor = DiagGaussianActor(obs_dim, action_dim, SAC_HIDDEN_DIM,
                              SAC_HIDDEN_DEPTH, SAC_LOG_STD_BOUNDS)
    actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    actor.to(ptu.device).eval()
    cache[task_idx] = actor
    return actor


def collect_context(algorithm, actor, task_j, n_steps):
    """Run `actor` inside task j and accumulate the transitions as context."""
    env = algorithm.env
    agent = algorithm.agent
    env.reset_task(task_j)
    agent.clear_z()
    o = env.reset()
    steps = 0
    while steps < n_steps:
        with torch.no_grad():
            dist = actor(torch.as_tensor(o, dtype=torch.float32,
                                         device=ptu.device).unsqueeze(0))
            a = dist.sample().clamp(-1, 1).squeeze(0).cpu().numpy()
        next_o, r, d, env_info = env.step(a)
        if callable(getattr(env, "sparsify_rewards", None)):
            env_info = {"sparse_reward": env.sparsify_rewards(r)}
        agent.update_context([o, a, r, next_o, d, env_info])
        o = next_o
        steps += 1
        if d:
            o = env.reset()
    agent.infer_posterior(agent.context)


def score_rollout(algorithm, task_j, max_path_length):
    """Deterministic meta-policy rollout in task j under the current posterior."""
    env = algorithm.env
    agent = algorithm.agent
    o = env.reset()
    ret = 0.0
    for _ in range(max_path_length):
        a, _ = agent.get_action(o, deterministic=True)
        o, r, d, _ = env.step(a)
        ret += float(r)
        if d:
            break
    return ret


def evaluate(algorithm, cfg, data_dir, n_policies, rng, context_steps=None):
    """Returns (ood_mean, iid_mean, n_pairs). Averages over test tasks and,
    for each, `n_policies` behavior policies drawn from the other tasks."""
    obs_dim = int(np.prod(algorithm.env.observation_space.shape))
    action_dim = int(np.prod(algorithm.env.action_space.shape))
    mpl = algorithm.max_path_length
    ctx_steps = context_steps or mpl
    cache = {}
    all_tasks = list(algorithm.env.get_all_task_idx())
    test_tasks = list(algorithm.train_tasks)

    ood, iid = [], []
    for j in test_tasks:
        others = [i for i in all_tasks if i != j]
        picks = rng.choice(others, size=min(n_policies, len(others)), replace=False)
        for i in picks:
            actor = load_behavior_actor(data_dir, int(i), obs_dim, action_dim, cache)
            collect_context(algorithm, actor, j, ctx_steps)
            ood.append(score_rollout(algorithm, j, mpl))
        # in-distribution reference: the task's own behavior policy
        actor = load_behavior_actor(data_dir, int(j), obs_dim, action_dim, cache)
        collect_context(algorithm, actor, j, ctx_steps)
        iid.append(score_rollout(algorithm, j, mpl))
    return float(np.mean(ood)), float(np.mean(iid)), len(ood)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, choices=ENV_GROUPS)
    p.add_argument("--algo", choices=list(ALGO_DIR))
    p.add_argument("--seed", type=int)
    p.add_argument("--all-runs", action="store_true",
                   help="evaluate every algo x seed found for --env")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--algos", default=",".join(ALGO_DIR),
                   help="with --all-runs: which algorithms to cover")
    p.add_argument("--ckpt-root", default="./output/checkpoints")
    p.add_argument("--data-root", default="./offline_dataset")
    p.add_argument("--n-policies", type=int, default=10,
                   help="OOD behavior policies sampled per test task")
    p.add_argument("--context-steps", type=int, default=None,
                   help="context transitions per evaluation (default: episode length)")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", default=None,
                   help="comma list or start:stop:step of training epochs to "
                        "score from agent_itr_<N>.pth, e.g. 0:95:10. Produces "
                        "a learning curve; omit to score agent.pth only.")
    args = p.parse_args()

    epochs = None
    if args.epochs:
        if ":" in args.epochs:
            a, b, c = (int(x) for x in args.epochs.split(":"))
            epochs = list(range(a, b + 1, c))
        else:
            epochs = [int(x) for x in args.epochs.split(",")]

    if args.all_runs:
        algos = [a.strip() for a in args.algos.split(",") if a.strip()]
        # variant exp_names (laco-a0.5, laco-auto, ...) map to themselves and are
        # launched with algo=laco plus the overrides that produced them
        for a in algos:
            ALGO_DIR.setdefault(a, a)
        runs = [(a, s) for a in algos for s in
                (int(x) for x in args.seeds.split(","))]
    else:
        if args.algo is None or args.seed is None:
            p.error("give --algo and --seed, or --all-runs")
        runs = [(args.algo, args.seed)]

    out_path = args.out or os.path.join(REPO, "plots",
                                        f"ood_behavior_{args.env}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # resume: keep rows already in the CSV and skip re-evaluating them
    rows, done = [], set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            rows = list(csv.DictReader(f))
        done = {(r["algo"], int(r["seed"]), int(r.get("epoch", -1))) for r in rows}
        if done:
            print(f"[resume] {len(done)} row(s) already in {out_path}")

    FIELDS = ["env", "algo", "seed", "epoch", "ood_behavior", "iid_behavior",
              "delta", "n_pairs", "n_policies", "minutes"]

    def flush():
        # rows may mix the pre-`epoch` schema with the current one, so always
        # write the full field set rather than inferring it from rows[0]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, -1 if k == "epoch" else "") for k in FIELDS})

    for algo, seed in runs:
        want = [(e, f"agent_itr_{e}.pth") for e in epochs] if epochs \
            else [(-1, "agent.pth")]
        want = [(e, f) for e, f in want if (ALGO_DIR[algo], seed, e) not in done]
        if not want:
            continue
        ck_dir = os.path.join(args.ckpt_root, ENV_NAME[args.env],
                              ALGO_DIR[algo], f"seed{seed}")
        if not os.path.isdir(ck_dir):
            print(f"[SKIP] no checkpoint dir: {ck_dir}")
            continue
        # the offline buffers dominate setup, so build once and re-load weights
        # exp_name may carry a variant suffix (laco-a0.5, unicorn-off, ...);
        # the hydra algo group is the base name before the first dash
        hydra_algo = algo.split("-")[0]
        hydra_algo = {"csro": "csro"}.get(hydra_algo, hydra_algo)
        algorithm, cfg = build_algorithm(args.env, hydra_algo, args.gpu_id)
        env_name = cfg.env_name
        data_dir = os.path.join(args.data_root, env_name)
        for epoch, fname in want:
            ckpt = os.path.join(ck_dir, fname)
            if not os.path.exists(ckpt):
                print(f"[SKIP] missing {ckpt}")
                continue
            t0 = time.time()
            algorithm.load_epoch_snapshot(ckpt)
            algorithm.training_mode(False)
            rng = np.random.default_rng(args.eval_seed)
            ood, iid, n = evaluate(algorithm, cfg, data_dir,
                                   args.n_policies, rng, args.context_steps)
            rows.append(dict(env=env_name, algo=ALGO_DIR[algo], seed=seed,
                             epoch=epoch, ood_behavior=round(ood, 2),
                             iid_behavior=round(iid, 2), delta=round(ood - iid, 2),
                             n_pairs=n, n_policies=args.n_policies,
                             minutes=round((time.time() - t0) / 60, 1)))
            print(f"[OK] {env_name} {ALGO_DIR[algo]} seed{seed} ep{epoch}: "
                  f"OOD={ood:.1f}  IID={iid:.1f}  ({rows[-1]['minutes']} min)")
            flush()
    print(f"\nwrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
