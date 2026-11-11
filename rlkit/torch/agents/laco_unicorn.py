"""LACO on the UNICORN-SS backbone (rebuttal experiment E2, 2026-10-21).

Reviewer 1 asked whether LACO is complementary to a stronger context encoder
such as UNICORN; Reviewers 3 and 4 asked that the "backbone-agnostic" claim be
backed by more than the CSRO backbone. This class is UNICORN-SS with the three
LACO terms attached to its actor--critic losses, exactly as `LACO` attaches
them to `CSROPrediction`:

  1. L_LC      latent-conservative critic penalty (log-sum-exp over the
               cross/interpolated/extrapolated proposal grid built from the
               other tasks of the meta-batch)
  2. L_Lip     critic z-smoothness (finite-radius surrogate)
  3. L_Lip_pi  policy z-smoothness (KL between pre-tanh Gaussians)

The encoder objective (L_recon + alpha_ratio * L_FOCAL) is untouched, the
latent is detached before the actor--critic update as in the backbone, and
with use_lc = use_lip = use_lip_pi = False this class reproduces UNICORN
exactly. The LACO helper methods are shared with `LACO` (bound below) so the
two backbones run bit-identical penalty code.
"""
import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.agents.unicorn import UNICORN
from rlkit.torch.agents.laco import LACO


class LACOUnicorn(UNICORN):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            extreme_tasks,
            latent_dim,
            nets,
            **kwargs,
        ):
        super().__init__(
            env=env,
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            extreme_tasks=extreme_tasks,
            latent_dim=latent_dim,
            nets=nets,
            **kwargs,
        )
        # --- LACO coefficients (same keys / defaults as LACO.__init__) ---
        self.alpha_lc     = float(kwargs.get('alpha_lc', 1.0))
        self.beta_lip     = float(kwargs.get('beta_lip', 1.0))
        self.kappa        = float(kwargs.get('kappa', 0.5))
        self.eta          = float(kwargs.get('eta', 0.5))
        self.use_lc       = bool(kwargs.get('use_lc', True))
        self.use_lip      = bool(kwargs.get('use_lip', True))
        self.use_lip_pi   = bool(kwargs.get('use_lip_pi', True))
        self.lc_use_cross = bool(kwargs.get('lc_use_cross', True))
        self.lc_use_intp  = bool(kwargs.get('lc_use_intp', True))
        self.lc_use_extp  = bool(kwargs.get('lc_use_extp', True))
        self.lc_tau       = float(kwargs.get('lc_tau', 1.0))
        self.lc_auto      = bool(kwargs.get('lc_auto', False))
        self.lc_target    = float(kwargs.get('lc_target', 0.9))
        self.lc_auto_lr   = float(kwargs.get('lc_auto_lr', 3e-4))
        self.lc_alpha_min = float(kwargs.get('lc_alpha_min', 0.01))
        self.lc_alpha_max = float(kwargs.get('lc_alpha_max', 100.0))
        self.lc_auto_warmup   = int(kwargs.get('lc_auto_warmup', 0))
        self.lc_auto_ema      = float(kwargs.get('lc_auto_ema', 0.99))
        self.lc_auto_max_step = float(kwargs.get('lc_auto_max_step', float('inf')))
        self._log_alpha_lc = float(np.log(max(self.alpha_lc, 1e-8)))
        self._c_lc_ema = None

    # The penalty code is literally LACO's: bind the helpers so any fix there
    # applies here too.
    _lc_proposals        = LACO._lc_proposals
    _lc_penalty          = LACO._lc_penalty
    _alpha_lc_effective  = LACO._alpha_lc_effective
    _sample_latent_noise = LACO._sample_latent_noise
    _critic_lip          = LACO._critic_lip

    # ------------------------------------------------------------------
    # Training step: verbatim UNICORN._take_step + marked LACO blocks
    # ------------------------------------------------------------------
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
        # UNICORN-SS encoder update (unchanged backbone objective)
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

        # === LACO: detached latents (same detach point as the backbone critic) ===
        loss_lc = None
        loss_lip = None
        hat_Lz = None
        loss_lip_pi = None
        if self.use_lc or self.use_lip or self.use_lip_pi:
            z_flat_det = task_z.detach()                              # [t*b, d]
            z_tasks_det = z_flat_det.view(t, b, -1)[:, 0, :]          # [t, d] one z per task
        if self.use_lc or self.use_lip:
            if self.allow_backward_z:
                q1_data = self.qf1(t, b, obs_flat, actions_flat, z_flat_det)
                q2_data = self.qf2(t, b, obs_flat, actions_flat, z_flat_det)
            else:
                q1_data, q2_data = q1_pred, q2_pred                   # already detached-z forwards

        # qf update
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat_scaled = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat_scaled + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)

        # === LACO: critic regularizers added on top of the TD loss ===
        qf_total = qf_loss
        if self.use_lc:
            loss_lc = self._lc_penalty(t, b, obs_flat, actions_flat, z_tasks_det, q1_data, q2_data)
            if loss_lc is not None:
                alpha_eff = self._alpha_lc_effective(loss_lc, qf_loss)
                qf_total = qf_total + alpha_eff * loss_lc
        if self.use_lip:
            loss_lip, hat_Lz = self._critic_lip(t, b, obs_flat, actions_flat, z_flat_det, z_tasks_det, q1_data, q2_data)
            qf_total = qf_total + self.beta_lip * loss_lip

        qf_total.backward(retain_graph=True)
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

        # === LACO: policy smoothness in z (actor loss) ===
        if self.use_lip_pi:
            delta_pi, delta_pi_sq = self._sample_latent_noise(z_flat_det, z_tasks_det)
            pert_in = torch.cat([obs_flat, z_flat_det + delta_pi], dim=1)
            pert_out = self.agent.policy(t, b, pert_in, deterministic=True)
            mean_pert, log_std_pert = pert_out[1], pert_out[2]
            var_ref = (2. * policy_log_std).exp()
            var_pert = (2. * log_std_pert).exp()
            kl = (log_std_pert - policy_log_std
                  + (var_ref + (policy_mean - mean_pert).pow(2)) / (2. * var_pert)
                  - 0.5).sum(dim=-1, keepdim=True)                    # [t*b, 1]
            loss_lip_pi = (kl / delta_pi_sq).mean()
            policy_loss = policy_loss + self.beta_lip * loss_lip_pi

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

        # === LACO stats (weighted contributions + raw diagnostics) ===
        if loss_lc is not None:
            a_eff = float(np.exp(self._log_alpha_lc)) if self.lc_auto else self.alpha_lc
            self.loss["loss_lc"] = (a_eff * loss_lc).item()
            self.eval_statistics['train/loss_lc'] = (a_eff * loss_lc).item()
            self.eval_statistics['train/lc_logsumexp_gap'] = loss_lc.item()
            self.eval_statistics['train/alpha_lc_eff'] = a_eff
            self.eval_statistics['train/C_lc'] = (
                a_eff * loss_lc.item() / max(float(qf_loss.item()) ** 0.5, 1e-6))
        if loss_lip is not None:
            self.loss["loss_lip"] = (self.beta_lip * loss_lip).item()
            self.eval_statistics['train/loss_lip'] = (self.beta_lip * loss_lip).item()
            self.eval_statistics['train/hat_Lz'] = hat_Lz.item()
        if loss_lip_pi is not None:
            self.loss["loss_lip_pi"] = (self.beta_lip * loss_lip_pi).item()
            self.eval_statistics['train/loss_lip_pi'] = (self.beta_lip * loss_lip_pi).item()

        return ptu.get_numpy(self.agent.z_means), ptu.get_numpy(self.agent.z_vars)
