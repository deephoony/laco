import os
from turtle import position
import torch
import torch.optim as optim
import numpy as np
import rlkit.torch.pytorch_util as ptu
from torch import nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from itertools import product
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import OfflineMetaRLAlgorithm
from rlkit.torch.brac import divergences
from rlkit.torch.brac import utils
from rlkit.torch.agents.agent import PEARLAgent
import pdb

class UNICORNMLPEncoder(nn.Module):
    """Official-style MLP encoder on (s, a, r, s', done) with stochastic samples."""

    def __init__(
            self,
            obs_dim,
            action_dim,
            latent_dim,
            hidden_size=64,
            num_hidden_layers=2,
            stochastic=True,
    ):
        super().__init__()
        self.output_size = latent_dim
        self.stochastic = stochastic

        in_dim = 2 * obs_dim + action_dim + 1 + 1
        layers = []
        cur_dim = in_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(cur_dim, hidden_size))
            layers.append(nn.ReLU())
            cur_dim = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(cur_dim, latent_dim)

        if self.stochastic:
            self.log_var = nn.Parameter(torch.zeros(1, latent_dim), requires_grad=True)

    def forward(self, context):
        original_shape = context.shape[:-1]
        flat_context = context.reshape(-1, context.shape[-1])
        out = self.head(self.backbone(flat_context))

        if self.stochastic:
            var = self.log_var.expand_as(out).exp()
            var = torch.clamp(var, 0.1, 10.0)
            out = torch.distributions.Normal(out, var.sqrt()).rsample()

        out = torch.tanh(out)
        return out.reshape(*original_shape, self.output_size)


class UNICORNAgent(PEARLAgent):
    """PEARL-style wrapper with UNICORN context layout: (s, a, r, s', done)."""

    def update_context(self, inputs):
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']

        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        if len(np.shape(r)) == 0:
            r = ptu.from_numpy(np.array([r], dtype=np.float32)[None, None, ...])
        else:
            r = ptu.from_numpy(np.array(r, dtype=np.float32)[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])
        d = ptu.from_numpy(np.array([d], dtype=np.float32)[None, None, ...])

        data = torch.cat([o, a, r, no, d], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def update_context_dict(self, batch_dict, env):
        o = ptu.from_numpy(batch_dict['observations'][None, ...])
        a = ptu.from_numpy(batch_dict['actions'][None, ...])
        next_o = ptu.from_numpy(batch_dict['next_observations'][None, ...])
        d = ptu.from_numpy(batch_dict['terminals'][None, ...].astype(np.float32))

        if callable(getattr(env, "sparsify_rewards", None)) and self.sparse_rewards:
            rewards = []
            for reward in batch_dict['rewards']:
                rewards.append(env.sparsify_rewards(reward))
            r = ptu.from_numpy(np.array(rewards, dtype=np.float32)[None, ...])
        else:
            r = ptu.from_numpy(batch_dict['rewards'][None, ...].astype(np.float32))

        data = torch.cat([o, a, r, next_o, d], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)


class UNICORN(OfflineMetaRLAlgorithm):  # UNICORN-SS: L_recon (Gaussian NLL on raw delta/reward) + alpha_ratio * L_FOCAL
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            extreme_tasks,
            latent_dim,
            nets,
            goal_radius=1,
            optimizer_class=optim.Adam,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            extreme_tasks=extreme_tasks,
            goal_radius=goal_radius,
            **kwargs
        )

        self.latent_dim                     = latent_dim
        self.soft_target_tau                = kwargs['soft_target_tau']
        self.policy_mean_reg_weight         = kwargs['policy_mean_reg_weight']
        self.policy_std_reg_weight          = kwargs['policy_std_reg_weight']
        self.policy_pre_activation_weight   = kwargs['policy_pre_activation_weight']
        self.recurrent                      = kwargs['recurrent']
        self.kl_lambda                      = kwargs['kl_lambda']
        self._divergence_name               = kwargs['divergence_name']
        self.sparse_rewards                 = kwargs['sparse_rewards']
        self.use_next_obs_in_context        = kwargs['use_next_obs_in_context']
        self.use_brac                       = kwargs['use_brac']
        self.use_value_penalty              = kwargs['use_value_penalty']
        self.alpha_max                      = kwargs['alpha_max']
        self._c_iter                        = kwargs['c_iter']
        self.train_alpha                    = kwargs['train_alpha']
        self._target_divergence             = kwargs['target_divergence']
        self.alpha_init                     = kwargs['alpha_init']
        self.alpha_lr                       = kwargs['alpha_lr']
        self.policy_lr                      = kwargs['policy_lr']
        self.qf_lr                          = kwargs['qf_lr']
        self.vf_lr                          = kwargs['vf_lr']
        self.c_lr                           = kwargs['c_lr']
        self.context_lr                     = kwargs['context_lr']
        self.z_loss_weight                  = kwargs['z_loss_weight']  # unused; kept for config compat
        self.max_entropy                    = kwargs['max_entropy']
        self.allow_backward_z               = kwargs['allow_backward_z']

        # UNICORN-SS (paper Eq.): L = L_recon + (alpha / (1-alpha)) * L_FOCAL
        # alpha_ratio == alpha/(1-alpha); corresponds to `beta_encoder` in the official repo.
        self.use_FOCAL_cl                   = kwargs.get('use_FOCAL_cl', True)
        self.alpha_ratio                    = kwargs.get('alpha_ratio', 0.15)
        self.decoder_lr                     = kwargs.get('decoder_lr', 0.0003)

        self.loss                           = {}
        self.plotter                        = plotter
        self.render_eval_paths              = render_eval_paths
        self.qf_criterion                   = nn.MSELoss()
        self.vf_criterion                   = nn.MSELoss()
        self.vib_criterion                  = nn.MSELoss()
        self.l2_reg_criterion               = nn.MSELoss()
        self.cross_entropy_loss             = nn.CrossEntropyLoss()
        self.mse_criterion                  = nn.MSELoss()

        self.qf1, self.qf2, self.vf, self.c, self.decoder = nets[1:]
        self.target_vf                      = self.vf.copy()

        self.policy_optimizer               = optimizer_class(self.agent.policy.parameters(), lr=self.policy_lr)
        self.qf1_optimizer                  = optimizer_class(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer                  = optimizer_class(self.qf2.parameters(), lr=self.qf_lr)
        self.vf_optimizer                   = optimizer_class(self.vf.parameters(),  lr=self.vf_lr)
        self.c_optimizer                    = optimizer_class(self.c.parameters(),   lr=self.c_lr)
        self.context_optimizer              = optimizer_class(self.agent.context_encoder.parameters(), lr=self.context_lr)
        self.decoder_optimizer              = optimizer_class(self.decoder.parameters(), lr=self.decoder_lr)

        self._num_steps                     = 0
        self._visit_num_steps_train         = 10
        self._alpha_var                     = torch.tensor(1.)

        for net in nets:
            self.print_networks(net)

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.qf1, self.qf2, self.vf, self.target_vf, self.c, self.decoder]

    @property
    def get_alpha(self):
        return utils.clip_v2(
            self._alpha_var, 0.0, self.alpha_max)

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
        if self.train_alpha:
            self._alpha_var = torch.tensor(self.alpha_init, device=ptu.device, requires_grad=True)
        self._divergence = divergences.get_divergence(name=self._divergence_name, c=self.c, device=ptu.device)

    def print_networks(self, net):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample trajectory-style context from the replay buffer '''
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [
            ptu.np_to_pytorch_batch(
                self.replay_buffer.random_batch(
                    idx,
                    batch_size=self.embedding_batch_size,
                    sequence=False,
                )
            )
            for idx in indices
        ]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        return torch.cat(context, dim=2)

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size 
        num_updates = self.embedding_batch_size // mb_size

        context_batch = self.sample_context(indices)
        self.agent.clear_z(num_tasks=len(indices))

        z_means_lst = []
        z_vars_lst = []
        
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self.loss['step'] = self._num_steps
            z_means, z_vars = self._take_step(indices, context)
            self._num_steps += 1
            z_means_lst.append(z_means[None, ...])
            z_vars_lst.append(z_vars[None, ...])
            self.agent.detach_z()
            
        z_means = np.mean(np.concatenate(z_means_lst), axis=0)
        z_vars = np.mean(np.concatenate(z_vars_lst), axis=0)
        return z_means, z_vars

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _optimize_c(self, indices, context):
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
        policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        c_loss = self._divergence.dual_critic_loss(obs, new_actions.detach(), actions, task_z.detach())
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()

    def FOCAL_z_loss(self, indices, task_z, task_z_vars, b, epsilon=1e-3, threshold=0.999):
        pos_z_loss = 0.
        neg_z_loss = 0.
        pos_cnt = 0
        neg_cnt = 0
        for i in range(len(indices)):
            idx_i = i * b 
            for j in range(i+1, len(indices)):
                idx_j = j * b 
                if indices[i] == indices[j]:
                    pos_z_loss += torch.sqrt(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon)
                    pos_cnt += 1
                else:
                    neg_z_loss += 1/(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon * 100)
                    neg_cnt += 1
        return pos_z_loss/(pos_cnt + epsilon) +  neg_z_loss/(neg_cnt + epsilon)

    def _take_step(self, indices, context):
        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_in_context = context[:, :, obs_dim + action_dim].cpu().numpy()
        self.loss["non_sparse_ratio"] = len(reward_in_context[np.nonzero(reward_in_context)]) / np.size(reward_in_context)

        num_tasks = len(indices)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)

        t, b, _ = obs.size()
        obs_flat = obs.view(t * b, -1)
        actions_flat = actions.view(t * b, -1)
        rewards_flat = rewards.view(t * b, -1)
        next_obs_flat = next_obs.view(t * b, -1)

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # Q and V networks
        if self.allow_backward_z:
            q1_pred = self.qf1(t, b, obs_flat, actions_flat, task_z)
            q2_pred = self.qf2(t, b, obs_flat, actions_flat, task_z)
            v_pred = self.vf(t, b, obs_flat, task_z.detach())
        else:
            q1_pred = self.qf1(t, b, obs_flat, actions_flat, task_z.detach())
            q2_pred = self.qf2(t, b, obs_flat, actions_flat, task_z.detach())
            v_pred = self.vf(t, b, obs_flat, task_z.detach())

        # BRAC
        c_loss = self._divergence.dual_critic_loss(obs_flat, new_actions.detach(), actions_flat, task_z.detach())
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()
        for _ in range(self._c_iter - 1):
            self._optimize_c(indices=indices, context=context)
        self.loss["c_loss"] = c_loss.item()

        div_estimate = self._divergence.dual_estimate(obs_flat, new_actions, actions_flat, task_z.detach())
        self.loss["div_estimate"] = torch.mean(div_estimate).item()

        with torch.no_grad():
            if self.use_brac and self.use_value_penalty:
                target_v_values = self.target_vf(t, b, next_obs_flat, task_z) - self.get_alpha * div_estimate
            else:
                target_v_values = self.target_vf(t, b, next_obs_flat, task_z)
        self.loss["target_v_values"] = torch.mean(target_v_values).item()

        # ==========================================================
        # UNICORN-SS (paper Eq.): L_encoder = L_recon + (alpha/(1-alpha)) * L_FOCAL
        # Faithful port of finetune_update_encoder in the official repo:
        #   - L_recon  : Gaussian NLL on raw (target_state - state) + raw target_reward,
        #                via FOCALGaussianDecoder (dynamic + reward GaussianMLP heads).
        #   - L_FOCAL  : FOCAL metric_loss.
        # ==========================================================
        self.context_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        recon_loss = self.decoder.loss(
            task_embedding=task_z,
            state=obs_flat,
            action=actions_flat,
            target_state=next_obs_flat,
            target_reward=rewards_flat,
        )

        focal_loss = self.FOCAL_z_loss(indices=indices, task_z=task_z, task_z_vars=task_z_vars, b=b)

        encoder_loss = recon_loss + self.alpha_ratio * focal_loss
        encoder_loss.backward(retain_graph=True)
        self.decoder_optimizer.step()
        self.context_optimizer.step()

        self.loss["recon_loss"] = recon_loss.item()
        self.loss["z_loss"] = focal_loss.item() if torch.is_tensor(focal_loss) else float(focal_loss)
        # ==========================================================

        # qf update
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat_scaled = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat_scaled + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)
        self.loss["qf_loss"] = qf_loss.item()
        self.loss["q_target"] = torch.mean(q_target).item()
        self.loss["q1_pred"] = torch.mean(q1_pred).item()
        self.loss["q2_pred"] = torch.mean(q2_pred).item()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = torch.min(self.qf1(t, b, obs_flat, new_actions, task_z.detach()),
                                        self.qf2(t, b, obs_flat, new_actions, task_z.detach()))

        # vf update
        if self.max_entropy:
            v_target = min_q_new_actions - log_pi
        else:
            v_target = min_q_new_actions
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward(retain_graph=True)
        self.vf_optimizer.step()
        self._update_target_network()
        self.loss["vf_loss"] = vf_loss.item()
        self.loss["v_target"] = torch.mean(v_target).item()
        self.loss["v_pred"] = torch.mean(v_pred).item()

        # policy update
        log_policy_target = min_q_new_actions

        if self.use_brac:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target + self.get_alpha.detach() * div_estimate).mean()
            else:
                policy_loss = (-log_policy_target + self.get_alpha.detach() * div_estimate).mean()
        else:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target).mean()
            else:
                policy_loss = - log_policy_target.mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=-1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.loss["policy_loss"] = policy_loss.item()
        
        # BRAC for training alpha:
        a_loss = -torch.mean(self._alpha_var * (div_estimate - self._target_divergence).detach())
        a_loss.backward()
        with torch.no_grad():
            self._alpha_var -= self.alpha_lr * self._alpha_var.grad
            self._alpha_var.grad.zero_()
        self.loss["a_loss"] = a_loss.item()

        for i in range(len(self.agent.z_means[0])):
            z_mean = ptu.get_numpy(self.agent.z_means[0][i])
            name = f'train/z_mean_{i}'
            self.eval_statistics[name] = z_mean

        z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))

        self.eval_statistics['train/Z_variance'] = z_sig
        self.eval_statistics['train/loss_focal'] = focal_loss.item() if torch.is_tensor(focal_loss) else float(focal_loss)
        self.eval_statistics['train/loss_recon'] = recon_loss.item()
        self.eval_statistics['train/alpha_ratio'] = self.alpha_ratio

        self.eval_statistics['train/loss_qf'] = np.mean(ptu.get_numpy(qf_loss))
        self.eval_statistics['train/loss_vf'] = np.mean(ptu.get_numpy(vf_loss))
        self.eval_statistics['train/loss_policy'] = np.mean(ptu.get_numpy(
            policy_loss
        ))
        if self.use_brac:
            self.eval_statistics['train/loss_dual_brack'] = np.mean(ptu.get_numpy(c_loss))
        self.eval_statistics['train/avg_q_values'] =  ptu.get_numpy(q1_pred).mean()
        self.eval_statistics['train/avg_v_values'] =  ptu.get_numpy(v_pred).mean()
        self.eval_statistics['train/log_policy'] =  ptu.get_numpy(log_pi).mean()
        self.eval_statistics['train/alpha'] =  ptu.get_numpy((self._alpha_var).reshape(-1)).mean()
        self.eval_statistics['train/div_estimate'] =  ptu.get_numpy(div_estimate).mean()
        
        return ptu.get_numpy(self.agent.z_means), ptu.get_numpy(self.agent.z_vars)

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            decoder=self.decoder.state_dict(),
            c=self.c.state_dict(),
            )
        return snapshot
