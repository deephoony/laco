import pathlib
import re
import numpy as np
import torch
import random
import hydra
import omegaconf
import os 
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.agents.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, EnsembleFlattenMlp
from rlkit.torch.agents.csro import CSROPrediction
from rlkit.torch.agents.laco import LACO
from rlkit.torch.agents.er_trl import ERTRL
from rlkit.torch.agents.unicorn import UNICORN, UNICORNAgent, UNICORNMLPEncoder
from rlkit.torch.agents.laco_unicorn import LACOUnicorn
from rlkit.torch.agents.unicorn_decoder import FOCALGaussianDecoder
from rlkit.torch.agents.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
import pdb

def global_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def experiment(algorithm, cfg):
    DEBUG = cfg.util_params.debug
    os.environ['DEBUG'] = str(int(DEBUG))
    
    exp_id = 'debug' if DEBUG else cfg.util_params.exp_name

    experiment_log_dir, wandb_logger = setup_logger(
        cfg.env_name,
        variant= omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        exp_id=exp_id,
        base_log_dir=cfg.util_params.base_log_dir, 
        seed=cfg.seed,
        snapshot_mode="gap_and_last",
        snapshot_gap=5, 
        use_wandb=cfg.use_wandb,
    )

    # optionally save eval trajectories as pkl files
    if cfg.algo_params.dump_eval_paths:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # optionally continue a killed run from its last saved weights
    start_epoch = 0
    resume_from = cfg.util_params.get('resume_from', None)
    if resume_from:
        resume_epoch = cfg.util_params.get('resume_epoch', None)
        if resume_epoch is None:
            m = re.search(r'agent_itr_(\d+)\.pth$', str(resume_from))
            if m is None:
                raise ValueError(
                    "util_params.resume_from=%s: cannot infer the epoch from "
                    "this filename, pass util_params.resume_epoch=<N> too "
                    "(N = last completed epoch)" % resume_from)
            resume_epoch = int(m.group(1))
        algorithm.load_epoch_snapshot(resume_from)
        start_epoch = int(resume_epoch) + 1
        print("[resume] restored %s, continuing at epoch %d/%s"
              % (resume_from, start_epoch,
                 cfg.algo_params.get('num_iterations', '?')))

    # run the algorithm
    algorithm.train(wandb_logger, start_epoch=start_epoch)
    if wandb_logger:
        wandb_logger.finish()

def initialize(cfg, ):
    if cfg.get("env_name") is None or cfg.get("algo_type") is None:
        raise ValueError(
            "Both Hydra groups must be selected before launch: use "
            "`python launch_experiment.py env=<env> algo=<algo> ...`."
        )

    # Backward-compatible alias: several env configs still use meta_batch_size.
    if "meta_batch_size" in cfg.algo_params:
        with omegaconf.open_dict(cfg):
            cfg.algo_params.meta_batch = cfg.algo_params.meta_batch_size

    # create multi-task environment and sample tasks, normalize obs if provided with 'normalizer.npz'
    env = NormalizedBoxEnv(ENVS[cfg.env_name](**dict(cfg.env_params)))
    global_seed(cfg.seed)
    try:
        env.reset(seed=cfg.seed)
    except TypeError:
        env.seed(cfg.seed)

    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = cfg.latent_size

    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if cfg.algo_params.use_next_obs_in_context else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if cfg.algo_params.use_information_bottleneck else latent_dim
    net_size = cfg.net_size
    recurrent = cfg.algo_params.recurrent
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    if cfg.algo_type in ('UNICORN', 'LACO_UNICORN'):
        context_encoder = UNICORNMLPEncoder(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_size=64,
            num_hidden_layers=2,
            stochastic=True,
        )
        qf_hidden_sizes = [256, 256]
        policy_hidden_sizes = [256, 256]
    else:
        context_encoder = encoder_model(
            hidden_sizes=cfg.encoder_dims,
            input_size=context_encoder_input_dim,
            output_size=context_encoder_output_dim,
            output_activation=torch.tanh,
        )
        qf_hidden_sizes = [net_size, net_size, net_size]
        policy_hidden_sizes = [net_size, net_size, net_size]

    qf1 = FlattenMlp(
        hidden_sizes=qf_hidden_sizes,
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=qf_hidden_sizes,
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=qf_hidden_sizes,
        input_size=obs_dim + latent_dim,
        output_size=1,
    )

    policy = TanhGaussianPolicy(
        hidden_sizes=policy_hidden_sizes,
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    # critic network for divergence in dual form (see BRAC paper https://arxiv.org/abs/1911.11361)
    c = FlattenMlp(
        hidden_sizes=policy_hidden_sizes,
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1
    )

    if cfg.algo_type in ('UNICORN', 'LACO_UNICORN'):
        agent = UNICORNAgent(
            latent_dim,
            context_encoder,
            policy,
            **dict(cfg.algo_params)
        )
    else:
        agent = PEARLAgent(
            latent_dim,
            context_encoder,
            policy,
            **dict(cfg.algo_params)
        )
    task_modes = env.task_modes()
    
    if cfg.algo_type in ('UNICORN', 'LACO_UNICORN'):
        # Faithful port of UNICORN/models/decoder.py::FOCALDecoder:
        #   - two GaussianMLP heads (dynamic + reward), num_layers=2, hidden=128,
        #     ensemble_size=1 (official default), learn_logvar_bounds=True.
        #   - NLL on raw (next_obs - obs) and raw reward.
        decoder = FOCALGaussianDecoder(
            obs_size=obs_dim,
            action_size=action_dim,
            task_embedding_size=latent_dim,
            num_layers=2,
            hidden_size=128,
        )
        algo_cls = LACOUnicorn if cfg.algo_type == 'LACO_UNICORN' else UNICORN
        algorithm = algo_cls(
            env=env,
            train_tasks=task_modes['train'],
            eval_tasks=task_modes['moderate'],
            extreme_tasks=task_modes['extreme'],
            nets=[agent, qf1, qf2, vf, c, decoder],
            latent_dim=latent_dim,
            **dict(cfg.algo_params)
        )
    
    elif cfg.algo_type == 'CSROPrediction':
        if cfg.algo_params.use_club_sa:
            club_input_dim = obs_dim + action_dim
        else:
            club_input_dim = obs_dim + action_dim + reward_dim if cfg.algo_params.use_next_obs_in_context else obs_dim + action_dim

        club_model = encoder_model(
            hidden_sizes=cfg.encoder_dims,
            input_size=club_input_dim,
            output_size=latent_dim * 2,
            output_activation=torch.tanh,
        )

        decoder = FlattenMlp(
            hidden_sizes = [net_size, net_size],
            input_size = obs_dim + action_dim + latent_dim,
            output_size= obs_dim + 1,
        )

        algorithm = CSROPrediction(
            env=env,
            train_tasks=task_modes['train'],
            eval_tasks=task_modes['moderate'],
            extreme_tasks=task_modes['extreme'],
            nets=[agent, qf1, qf2, vf, c, club_model],
            latent_dim=latent_dim,
            decoder=decoder,
            **dict(cfg.algo_params)
        )

    elif cfg.algo_type == 'LACO':
        # LACO = CSRO backbone + latent-conservative / z-Lipschitz regularizers.
        # Network construction mirrors the CSROPrediction branch exactly.
        if cfg.algo_params.use_club_sa:
            club_input_dim = obs_dim + action_dim
        else:
            club_input_dim = obs_dim + action_dim + reward_dim if cfg.algo_params.use_next_obs_in_context else obs_dim + action_dim

        club_model = encoder_model(
            hidden_sizes=cfg.encoder_dims,
            input_size=club_input_dim,
            output_size=latent_dim * 2,
            output_activation=torch.tanh,
        )

        decoder = FlattenMlp(
            hidden_sizes = [net_size, net_size],
            input_size = obs_dim + action_dim + latent_dim,
            output_size= obs_dim + 1,
        )

        algorithm = LACO(
            env=env,
            train_tasks=task_modes['train'],
            eval_tasks=task_modes['moderate'],
            extreme_tasks=task_modes['extreme'],
            nets=[agent, qf1, qf2, vf, c, club_model],
            latent_dim=latent_dim,
            decoder=decoder,
            **dict(cfg.algo_params)
        )

    elif cfg.algo_type == 'ERTRL':
        generator = FlattenMlp(
            hidden_sizes = cfg.encoder_dims,
            input_size = latent_dim+obs_dim+cfg.algo_params.generator_dim,
            output_size = action_dim,
            output_activation = torch.tanh
         )
        discriminator = FlattenMlp(
            hidden_sizes = [net_size, net_size],
            input_size = action_dim+obs_dim+latent_dim,
            output_size = 1,
            output_activation = torch.sigmoid,
        )

        algorithm = ERTRL(
            env=env,
            train_tasks=task_modes['train'],
            eval_tasks=task_modes['moderate'],
            extreme_tasks=task_modes['extreme'],
            nets=[agent, qf1, qf2, vf, c, generator, discriminator],
            latent_dim=latent_dim,
            **dict(cfg.algo_params)
        )

    # optional GPU mode
    ptu.set_gpu_mode(cfg.use_gpu, cfg.gpu_id)
    algorithm.to()
    return algorithm


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@hydra.main(version_base="1.3", config_path="./cfgs", config_name="experiment")
def main(cfg):
    algorithm = initialize(cfg)
    experiment(algorithm, cfg)

if __name__ == "__main__":
    main()
