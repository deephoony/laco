"""Matched-behavior (IID) return with several independent contexts per task.

The behavior-OOD evaluation (scripts/eval_ood_behavior.py) averages 10 donor
policies x 20 test tasks = 200 context-rollout pairs per seed for the OOD
column, but only ONE context per task (20 pairs) for the IID column, so the IID
column carries roughly sqrt(10) times more sampling noise. This script
re-measures the IID column only, with `--repeats` independent contexts per test
task (default 10, i.e. 200 pairs per seed, the same budget as the OOD column).

Each context is collected by the test task's own converged SAC policy executed
inside that task (identical to the IID reference of eval_ood_behavior.py), and
a separate deterministic rollout is scored under the inferred latent. The
torch / numpy RNGs are seeded per run so the measurement is reproducible.

Resume-safe: rows already present in --out are skipped.

Usage:
  python scripts/eval_iid_repeat.py --env hopper-mass \
      --algos csro,laco-aut0.5 --seeds 0,1,2,3,4,5 \
      --ckpt-root ./output/checkpoints_bengio --gpu-id 0 \
      --out ./output/iid_hopper-mass.csv
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_ood_behavior import (ALGO_DIR, ENV_GROUPS, ENV_NAME,  # noqa: E402
                               build_algorithm, collect_context,
                               load_behavior_actor, score_rollout)

FIELDS = ["env", "algo", "seed", "epoch", "iid_rep", "iid_task_sd", "n_pairs",
          "repeats", "minutes", "per_task"]


def evaluate_iid(algorithm, data_dir, repeats, context_steps=None):
    obs_dim = int(np.prod(algorithm.env.observation_space.shape))
    action_dim = int(np.prod(algorithm.env.action_space.shape))
    mpl = algorithm.max_path_length
    ctx_steps = context_steps or mpl
    cache = {}
    per_task = {}
    for j in list(algorithm.train_tasks):
        actor = load_behavior_actor(data_dir, int(j), obs_dim, action_dim, cache)
        vals = []
        for _ in range(repeats):
            collect_context(algorithm, actor, j, ctx_steps)
            vals.append(score_rollout(algorithm, j, mpl))
        per_task[int(j)] = float(np.mean(vals))
    v = np.array(list(per_task.values()))
    return float(v.mean()), float(v.std()), len(v) * repeats, per_task


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, choices=ENV_GROUPS)
    p.add_argument("--algos", required=True)
    p.add_argument("--seeds", default="0,1,2,3,4,5")
    p.add_argument("--ckpt-root", required=True)
    p.add_argument("--data-root", default="./offline_dataset")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    for a in algos:
        ALGO_DIR.setdefault(a, a)
    seeds = [int(x) for x in args.seeds.split(",")]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows, done = [], set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            rows = list(csv.DictReader(f))
        done = {(r["algo"], int(r["seed"])) for r in rows}

    def flush():
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    for algo in algos:
        todo = [s for s in seeds if (ALGO_DIR[algo], s) not in done]
        todo = [s for s in todo if os.path.exists(os.path.join(
            args.ckpt_root, ENV_NAME[args.env], ALGO_DIR[algo], f"seed{s}", "agent.pth"))]
        if not todo:
            continue
        hydra_algo = algo.split("-")[0]
        algorithm, cfg = build_algorithm(args.env, hydra_algo, args.gpu_id)
        data_dir = os.path.join(args.data_root, cfg.env_name)
        for s in todo:
            ck = os.path.join(args.ckpt_root, ENV_NAME[args.env], ALGO_DIR[algo],
                              f"seed{s}", "agent.pth")
            t0 = time.time()
            algorithm.load_epoch_snapshot(ck)
            algorithm.training_mode(False)
            torch.manual_seed(10_000 + args.eval_seed)
            np.random.seed(10_000 + args.eval_seed)
            m, sd, n, per_task = evaluate_iid(algorithm, data_dir, args.repeats)
            rows.append(dict(env=cfg.env_name, algo=ALGO_DIR[algo], seed=s, epoch=-1,
                             iid_rep=round(m, 2), iid_task_sd=round(sd, 2), n_pairs=n,
                             repeats=args.repeats,
                             minutes=round((time.time() - t0) / 60, 1),
                             per_task=json.dumps({k: round(v, 2) for k, v in per_task.items()})))
            print(f"[OK] {cfg.env_name} {ALGO_DIR[algo]} seed{s}: IID x{args.repeats} = {m:.1f} "
                  f"({rows[-1]['minutes']} min)", flush=True)
            flush()
        del algorithm
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
