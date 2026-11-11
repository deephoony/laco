"""Inference latency of a trained COMRL agent (Reviewer 1, comment 4).

Measures, for each algorithm, on CPU (one thread) and on one GPU:
  * latent inference: encoder forward on a context of the size used at evaluation,
    plus the posterior computation (agent.infer_posterior), batch 1;
  * one decision: agent.get_action(obs, deterministic=True), batch 1.
Algorithms are measured in interleaved rounds so that they meet the same machine
state. Reports the median and the 10th-90th percentile over all calls.

  python scripts/bench_inference.py --env ant-dir --algos laco,csro --device cpu
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def build(env, algo, device):
    import eval_ood_behavior as E
    if device == "cpu":
        from hydra import compose, initialize_config_dir
        import launch_experiment
        with initialize_config_dir(config_dir=os.path.join(E.REPO, "cfgs"), version_base="1.3",
                                   job_name="bench"):
            cfg = compose(config_name="experiment",
                          overrides=[f"env={env}", f"algo={algo.split('-')[0]}", "use_wandb=false",
                                     "use_gpu=false", "gpu_id=0"])
        return launch_experiment.initialize(cfg), cfg
    return E.build_algorithm(env, algo.split("-")[0], 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="ant-dir")
    p.add_argument("--algos", default="laco,csro")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-root", default="./output/checkpoints_rev")
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--calls", type=int, default=200)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(1)
    import eval_ood_behavior as E
    import rlkit.torch.pytorch_util as ptu
    algos = a.algos.split(",")
    agents = {}
    for al in algos:
        algorithm, cfg = build(a.env, al, a.device)
        d = E.ALGO_DIR.get(al, al)
        algorithm.load_epoch_snapshot(os.path.join(a.ckpt_root, E.ENV_NAME[a.env], d,
                                                   f"seed{a.seed}", "agent.pth"))
        algorithm.training_mode(False)
        task = algorithm.train_tasks[0]
        ctx = algorithm.sample_context([task])
        o = algorithm.env.reset()
        agents[al] = (algorithm, ctx, o)
        n_params = sum(p_.numel() for m in (algorithm.agent.policy, algorithm.agent.context_encoder)
                       for p_ in m.parameters())
        print(f"{al}: context shape {tuple(ctx.shape)}, policy+encoder params {n_params}", flush=True)
    sync = (lambda: torch.cuda.synchronize()) if a.device == "gpu" else (lambda: None)
    res = {al: {"infer": [], "act": []} for al in algos}
    with torch.no_grad():
        for al in algos:                                   # warm-up
            algorithm, ctx, o = agents[al]
            for _ in range(20):
                algorithm.agent.infer_posterior(ctx)
                algorithm.agent.get_action(o, deterministic=True)
        for _ in range(a.rounds):
            for al in algos:
                algorithm, ctx, o = agents[al]
                for _ in range(a.calls // 10):
                    sync(); t0 = time.perf_counter()
                    algorithm.agent.infer_posterior(ctx)
                    sync(); res[al]["infer"].append(time.perf_counter() - t0)
                for _ in range(a.calls):
                    sync(); t0 = time.perf_counter()
                    algorithm.agent.get_action(o, deterministic=True)
                    sync(); res[al]["act"].append(time.perf_counter() - t0)
    out = {}
    for al in algos:
        out[al] = {k: dict(median_ms=1e3 * float(np.median(v)), p10_ms=1e3 * float(np.percentile(v, 10)),
                           p90_ms=1e3 * float(np.percentile(v, 90)), n=len(v)) for k, v in res[al].items()}
        print(al, json.dumps(out[al]))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"env": a.env, "device": a.device, "results": out}, f, indent=1)


if __name__ == "__main__":
    main()
