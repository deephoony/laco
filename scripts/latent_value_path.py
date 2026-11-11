"""Value along a straight path through latent space (Fig. 2 of the paper).

The claim LACO rests on is that the critic is unconstrained along its third
argument: outside the latent region the data supports, Q(s,a,z) may take any
value, and the policy that maximises it commits to whatever it finds there.
This script measures that surface directly.

For a fixed batch of state-action pairs drawn from task i's own dataset we hold
(s,a) fixed and sweep only the latent along

    z_lambda = lambda * z_i + (1 - lambda) * z_j ,   lambda in [-0.5, 1.5]

where z_i, z_j are the posterior means of two meta-training tasks. lambda in
(0,1) interpolates between two supported latents; lambda outside [0,1]
extrapolates past them and is exactly the region no training gradient visits.
Averaging Q over the (s,a) batch and over many (i,j) pairs gives one curve per
checkpoint, which is directly comparable across algorithms once each curve is
expressed relative to its own value at the supported endpoint lambda=1.

Two readouts are written per run:
  q_lambda   the curve itself, normalised by |Q(z_i)| so environments whose
             Bellman scale differs by two orders of magnitude are comparable
  blowup     max over the extrapolation region of Q(z_lambda) - Q(z_i), the
             single number that says how far the critic runs away off support

Usage:
  python scripts/latent_value_path.py --env ant-dir --algos csro,laco --seeds 0,1,2
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from eval_ood_behavior import (ALGO_DIR, ENV_GROUPS, ENV_NAME,  # noqa: E402
                               build_algorithm)
import rlkit.torch.pytorch_util as ptu  # noqa: E402

FIELDS = ["env", "algo", "seed", "lam", "q_rel", "q_raw", "q_sd"]


def task_latents(algorithm, tasks, n_ctx):
    """Posterior mean for each task from a batch of its own dataset transitions."""
    agent = algorithm.agent
    out = {}
    for t in tasks:
        idx = algorithm.train_tasks.index(t) if t in algorithm.train_tasks else None
        if idx is None:
            continue
        ctx = algorithm.sample_context([t])
        agent.clear_z()
        agent.infer_posterior(ctx)
        out[t] = agent.z.detach().clone().reshape(1, -1)
    return out


def sa_batch(algorithm, task, n):
    """A fixed batch of (s,a) from the task's own dataset."""
    b = algorithm.replay_buffer.random_batch(task, n)
    s = ptu.from_numpy(b["observations"])
    a = ptu.from_numpy(b["actions"])
    return s, a


def q_at(algorithm, s, a, z):
    """Mean of the twin critics at a single latent, broadcast over the batch.

    FlattenMlp.forward takes (meta_size, batch_size, *inputs); the two leading
    integers are bookkeeping and only the trailing tensors are concatenated, so
    a single task with the full batch is (1, n).
    """
    n = s.shape[0]
    zz = z.repeat(n, 1)
    with torch.no_grad():
        q1 = algorithm.qf1(1, n, s, a, zz)
        q2 = algorithm.qf2(1, n, s, a, zz)
    return torch.min(q1, q2).squeeze(-1).cpu().numpy()


def sweep(algorithm, lams, n_pairs, n_sa, rng):
    tasks = list(algorithm.train_tasks)
    zs = task_latents(algorithm, tasks, n_sa)
    keys = [t for t in tasks if t in zs]
    curves = []
    for _ in range(n_pairs):
        i, j = rng.choice(len(keys), size=2, replace=False)
        ti, tj = keys[i], keys[j]
        s, a = sa_batch(algorithm, ti, n_sa)
        zi, zj = zs[ti], zs[tj]
        # anchor: the value at the supported endpoint the batch belongs to
        q_anchor = float(np.mean(q_at(algorithm, s, a, zi)))
        row = []
        for lam in lams:
            z = lam * zi + (1.0 - lam) * zj
            row.append(float(np.mean(q_at(algorithm, s, a, z))))
        scale = max(abs(q_anchor), 1e-6)
        curves.append([(q - q_anchor) / scale for q in row])
    c = np.asarray(curves)
    return c.mean(axis=0), c.std(axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, choices=ENV_GROUPS)
    p.add_argument("--algos", default="csro,laco")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--ckpt-root", default="./output/checkpoints")
    p.add_argument("--lam-min", type=float, default=-0.5)
    p.add_argument("--lam-max", type=float, default=1.5)
    p.add_argument("--n-lam", type=int, default=41)
    p.add_argument("--n-pairs", type=int, default=40)
    p.add_argument("--n-sa", type=int, default=256)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    lams = np.linspace(args.lam_min, args.lam_max, args.n_lam)
    out_path = args.out or f"./output/latent_path/{args.env}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []

    for algo in args.algos.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            ck = os.path.join(args.ckpt_root, ENV_NAME[args.env],
                              ALGO_DIR.get(algo, algo), f"seed{seed}", "agent.pth")
            if not os.path.exists(ck):
                print(f"[SKIP] {ck}")
                continue
            hydra_algo = algo.split("-")[0]
            algorithm, _ = build_algorithm(args.env, hydra_algo, args.gpu_id)
            algorithm.load_epoch_snapshot(ck)
            algorithm.training_mode(False)
            rng = np.random.default_rng(0)
            m, sd = sweep(algorithm, lams, args.n_pairs, args.n_sa, rng)
            for lam, mm, ss in zip(lams, m, sd):
                rows.append(dict(env=ENV_NAME[args.env], algo=ALGO_DIR.get(algo, algo),
                                 seed=seed, lam=round(float(lam), 4),
                                 q_rel=round(float(mm), 5), q_raw="",
                                 q_sd=round(float(ss), 5)))
            ex = [mm for lam, mm in zip(lams, m) if lam < 0 or lam > 1]
            print(f"[OK] {args.env} {algo} seed{seed}: "
                  f"max off-support excess = {max(ex):+.3f} (relative)")
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            del algorithm
    print(f"wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
