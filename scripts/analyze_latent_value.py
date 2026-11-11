"""What the critic believes on a corrupted latent, and what it actually gets.

The paper's claim is that no offline meta-RL method guards its latent input: the
encoder is trained to identify tasks, the critic is trained to be conservative
in (s,a), and nothing constrains Q along z. Section 3's schematic asserts this.
This script measures it.

For each test task M^j the context is collected twice through the same offline
machinery, once by task j's own converged SAC policy (IID) and once by another
task's (UNICORN Sec. 3.3 behavior-OOD). Under the corrupted posterior the
meta-policy is rolled out, and for states along that rollout we record

  predicted   min(Q1,Q2)(s_t, a_t, z_OOD)     -- what the agent believes
  attained    sum_k gamma^k r_{t+k}           -- what the rollout actually pays

A critic that is honest on off-support latents sits on the diagonal. One that
extrapolates optimistically sits above it, and the vertical distance is the
overestimation the conservative term is supposed to remove.

Two cheaper quantities come free from the same rollout:

  sensitivity   mean |Q(s,a,z+d) - Q(s,a,z)| / (|Q| + eps) over ||d|| = eps
                -- the empirical Lipschitz profile L_Lip is meant to flatten
  displacement  ||z_OOD - z_IID||
                -- how far the encoder was actually moved. LACO does not touch
                the encoder objective, so this should be comparable across
                methods; if it is, any return difference is the critic's doing.

  python scripts/analyze_latent_value.py --env humanoid-dir --algo laco-auto \
      --seeds 0 1 2 3 4 5 --cpu --out ./output/latent_value/

Writes one CSV per (env, algo) so parallel workers never share a file.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import rlkit.torch.pytorch_util as ptu  # noqa: E402
from eval_ood_behavior import collect_context, load_behavior_actor  # noqa: E402

# the per-environment LACO configuration the paper reports
FINAL = {"ant-dir": "laco", "ant-goal": "laco-a4.2", "cheetah-vel": "laco",
         "humanoid-dir": "laco-auto", "hopper-mass": "laco-aut0.5",
         "walker-friction": "laco-auto"}
# ckpt directory name differs from the hydra env group for the four -custom ones
CKPT_ENV = {"ant-dir": "ant-dir-custom", "ant-goal": "ant-goal-custom",
            "cheetah-vel": "cheetah-vel-custom", "humanoid-dir": "humanoid-dir-custom",
            "hopper-mass": "hopper-mass", "walker-friction": "walker-friction"}
EPS_GRID = [0.25, 0.5, 1.0, 2.0]


def build(env_group, algo, gpu_id, cpu):
    """Reconstruct the training-time algorithm object on the requested device."""
    from hydra import compose, initialize_config_dir
    import launch_experiment
    ov = ["use_wandb=false"]
    ov += ["use_gpu=false"] if cpu else ["use_gpu=true", f"gpu_id={gpu_id}"]
    with initialize_config_dir(config_dir=os.path.join(REPO, "cfgs"),
                               version_base="1.3", job_name="latent_value"):
        cfg = compose(config_name="experiment",
                      overrides=[f"env={env_group}", f"algo={algo}", *ov])
    return launch_experiment.initialize(cfg), cfg


def rollout(algorithm, task_j, mpl):
    """Deterministic rollout under the current posterior; keep the transitions."""
    env, agent = algorithm.env, algorithm.agent
    env.reset_task(task_j)
    o = env.reset()
    obs, acts, rews = [], [], []
    for _ in range(mpl):
        a, _ = agent.get_action(o, deterministic=True)
        no, r, d, _ = env.step(a)
        obs.append(o); acts.append(a); rews.append(float(r))
        o = no
        if d:
            break
    return np.asarray(obs, np.float32), np.asarray(acts, np.float32), np.asarray(rews)


def discounted_to_go(rews, gamma):
    """G_t = sum_k gamma^k r_{t+k}, computed backwards in one pass."""
    g = np.empty_like(rews)
    acc = 0.0
    for i in range(len(rews) - 1, -1, -1):
        acc = rews[i] + gamma * acc
        g[i] = acc
    return g


def critic(algorithm, obs, acts, z):
    """min(Q1,Q2) for a batch of states under a single latent.

    FlattenMlp.forward swallows two positional placeholders (meta_size,
    batch_size) before the tensors it actually concatenates, so the critic must
    be called as qf(t, b, obs, act, z). The agents' own _min_q helper omits
    them and is dead code -- calling it silently feeds obs in as meta_size.
    """
    o = torch.as_tensor(obs, device=ptu.device)
    a = torch.as_tensor(acts, device=ptu.device)
    zz = z.expand(o.shape[0], -1)
    with torch.no_grad():
        q1 = algorithm.qf1(1, o.shape[0], o, a, zz)
        q2 = algorithm.qf2(1, o.shape[0], o, a, zz)
        return torch.min(q1, q2).squeeze(-1).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=list(FINAL))
    p.add_argument("--algo", required=True,
                   help="checkpoint variant: CSRO / unicorn / ertrl / a LACO config")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--n-tasks", type=int, default=10)
    p.add_argument("--n-states", type=int, default=40,
                   help="states sampled per rollout for the value comparison")
    p.add_argument("--ckpt-root", default="./output/checkpoints")
    p.add_argument("--data-root", default="./offline_dataset")
    p.add_argument("--out", default="./output/latent_value")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()

    base = a.algo.split("-")[0].lower()          # laco-auto -> laco, CSRO -> csro
    algorithm, cfg = build(a.env, base, a.gpu_id, a.cpu)
    mpl = algorithm.max_path_length
    gamma = float(algorithm.discount)
    rscale = float(getattr(algorithm, "reward_scale", 1.0))
    obs_dim = int(np.prod(algorithm.env.observation_space.shape))
    act_dim = int(np.prod(algorithm.env.action_space.shape))
    data_dir = os.path.join(a.data_root, cfg.env_name)
    tasks = list(algorithm.train_tasks)[: a.n_tasks]
    all_tasks = list(algorithm.env.get_all_task_idx())
    # one donor per task, fixed across methods and seeds so every method faces
    # the identical corrupted context and the comparison is like-for-like
    donors = {j: int(np.random.default_rng(1000 + j).choice(
        [i for i in all_tasks if i != j])) for j in tasks}

    rows, cache = [], {}
    for sd in a.seeds:
        ck = os.path.join(a.ckpt_root, CKPT_ENV[a.env], a.algo, f"seed{sd}", "agent.pth")
        if not os.path.exists(ck):
            print(f"  [SKIP] {ck}")
            continue
        algorithm.load_epoch_snapshot(ck)
        algorithm.training_mode(False)
        agent = algorithm.agent

        for j in tasks:
            own = load_behavior_actor(data_dir, j, obs_dim, act_dim, cache)
            collect_context(algorithm, own, j, mpl)
            z_iid = agent.z.detach().clone()

            donor = load_behavior_actor(data_dir, donors[j], obs_dim, act_dim, cache)
            collect_context(algorithm, donor, j, mpl)
            z_ood = agent.z.detach().clone()

            # roll out under each posterior in turn. The IID arm is what makes
            # the comparison a statement about the LATENT rather than about the
            # policy: both rollouts share weights and task, only z differs.
            got = {}
            for tag, z in (("ood", z_ood), ("iid", z_iid)):
                agent.z = z
                obs, acts, rews = rollout(algorithm, j, mpl)
                if len(rews) < 5:
                    break
                g = discounted_to_go(rews, gamma)
                # value of the initial state against what the episode actually
                # paid. Later states are truncated by the episode end and would
                # bias the comparison downwards, so only the head is used.
                n = min(a.n_states, max(1, len(rews) // 4))
                idx = np.arange(n)
                # Q is trained on rewards multiplied by reward_scale (5.0 for
                # CSRO/ER-TRL/LACO, 1.0 for UNICORN except ant-*), so the raw
                # critic output is NOT on the return scale. Undo it or the
                # comparison measures the config, not the critic.
                q = critic(algorithm, obs[idx], acts[idx], z) / rscale
                got[tag] = (float(np.mean(q)), float(np.mean(g[idx])),
                            float(np.sum(rews)), obs[idx], acts[idx])
            if len(got) < 2:
                continue

            sens = {}
            rng = np.random.default_rng(7 + sd)
            o_s, a_s = got["ood"][3], got["ood"][4]
            q_ref = critic(algorithm, o_s, a_s, z_ood) / rscale
            for eps in EPS_GRID:
                d = rng.normal(size=z_ood.shape).astype(np.float32)
                d /= np.linalg.norm(d)
                zp = z_ood + torch.as_tensor(d * eps, device=ptu.device)
                qp = critic(algorithm, o_s, a_s, zp) / rscale
                sens[eps] = float(np.mean(np.abs(qp - q_ref))
                                  / (np.mean(np.abs(q_ref)) + 1e-6))

            rows.append(dict(
                env=a.env, algo=a.algo, seed=sd, task=j, donor=donors[j],
                reward_scale=rscale,
                predicted=got["ood"][0], attained=got["ood"][1],
                undiscounted=got["ood"][2],
                predicted_iid=got["iid"][0], attained_iid=got["iid"][1],
                undiscounted_iid=got["iid"][2],
                # what the critic thinks it lost by being handed a wrong latent,
                # against what it actually lost
                dq=got["ood"][0] - got["iid"][0],
                dg=got["ood"][1] - got["iid"][1],
                displacement=float(torch.norm(z_ood - z_iid).item()),
                z_norm=float(torch.norm(z_iid).item()),
                **{f"sens_{e}": v for e, v in sens.items()}))
        done = [r for r in rows if r["seed"] == sd]
        if done:
            m = lambda k: np.mean([r[k] for r in done])
            print(f"  seed{sd} rs={rscale:g}  Q {m('predicted'):8.1f}  G {m('attained'):8.1f}"
                  f"  ratio {m('predicted')/(abs(m('attained'))+1e-6):6.2f}"
                  f"   dQ {m('dq'):+8.1f}  dG {m('dg'):+8.1f}"
                  f"   disp {m('displacement'):5.2f}")

    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"lv_{a.env}_{a.algo}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
