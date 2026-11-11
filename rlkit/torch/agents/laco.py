"""LACO: latent-conservative + latent-Lipschitz regularization on the CSRO backbone.

LACO = CSROPrediction (encoder objective, BRAC actor-critic, z-detach behavior
all unchanged) + three additive regularizers, every one computed on DETACHED
task latents (the same detach point the backbone already uses for its critic):

  1. L_LC      latent-conservative critic penalty. For each task i in the meta
               batch, build a proposal set N_i from the other tasks' latents
               (cross), interpolations lam*z_i + (1-lam)*z_j with lam~U(0,1)
               (intp) and extrapolations with lam~U(1, 1+eta) (extp), then
               penalize  mean[ logsumexp_{z~N_i} Q(s,a,z) - Q(s,a,z_i) ].
  2. L_Lip     critic smoothness in z:  mean[ (Q(s,a,z+d)-Q(s,a,z))^2 / ||d||^2 ]
               with d ~ N(0, sigma^2 I), sigma = kappa * per-dim std of the
               batch latents (std detached).
  3. L_Lip_pi  policy smoothness in z:  mean[ KL(pi(.|s,z) || pi(.|s,z+d)) / ||d||^2 ]
               with the analytic diagonal-Gaussian KL of the pre-tanh heads.

L_LC and L_Lip are added to the critic (qf1 AND qf2, averaged) loss with
coefficients alpha_lc and beta_lip; L_Lip_pi is added to the actor loss with
coefficient beta_lip.

With use_lc=use_lip=use_lip_pi=false this class performs exactly the same
computation (and the same RNG draws) as CSROPrediction. This file is purely
ADDITIVE: csro.py is untouched.
"""
import torch
import numpy as np
import rlkit.torch.pytorch_util as ptu
from torch import nn as nn
import torch.nn.functional as F
from rlkit.torch.agents.csro import CSROPrediction


class LACO(CSROPrediction):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            extreme_tasks,
            latent_dim,
            nets,
            decoder,
            **kwargs,
        ):
        super().__init__(
            env,
            train_tasks,
            eval_tasks,
            extreme_tasks,
            latent_dim,
            nets,
            decoder,
            **kwargs,
        )
        # --- LACO coefficients ---
        self.alpha_lc     = float(kwargs.get('alpha_lc', 1.0))
        self.beta_lip     = float(kwargs.get('beta_lip', 1.0))
        self.kappa        = float(kwargs.get('kappa', 0.5))
        self.eta          = float(kwargs.get('eta', 0.5))
        # --- ablation toggles (all false => exact CSRO behavior) ---
        self.use_lc       = bool(kwargs.get('use_lc', True))
        self.use_lip      = bool(kwargs.get('use_lip', True))
        self.use_lip_pi   = bool(kwargs.get('use_lip_pi', True))
        self.lc_use_cross = bool(kwargs.get('lc_use_cross', True))
        self.lc_use_intp  = bool(kwargs.get('lc_use_intp', True))
        self.lc_use_extp  = bool(kwargs.get('lc_use_extp', True))
        # --- logsumexp temperature -------------------------------------
        # tau*logsumexp(q/tau) interpolates between max (tau->0) and the mean
        # plus tau*log K (tau->inf). At tau=1 with |Q| ~ 50-70 and log K ~ 3.5
        # the soft-max is an effectively hard max over the single best
        # proposal, which is not the expectation the method is motivated by.
        # tau=1.0 reproduces the original behaviour exactly.
        self.lc_tau       = float(kwargs.get('lc_tau', 1.0))
        # --- optional dual control of alpha_lc --------------------------
        # alpha_lc is an absolute coefficient while the Bellman scale varies
        # ~100x across environments, so one value cannot serve all of them.
        # With lc_auto=True, alpha_lc is instead driven so that the effective
        # penalty ratio C_lc = alpha_lc * gap / sqrt(qf_loss) tracks
        # lc_target; both quantities are then in the same Q units.
        self.lc_auto      = bool(kwargs.get('lc_auto', False))
        self.lc_target    = float(kwargs.get('lc_target', 0.9))
        self.lc_auto_lr   = float(kwargs.get('lc_auto_lr', 3e-4))
        self.lc_alpha_min = float(kwargs.get('lc_alpha_min', 0.01))
        self.lc_alpha_max = float(kwargs.get('lc_alpha_max', 100.0))
        # Stability of the dual step. C_lc divides by sqrt(qf_loss), so early in
        # training -- when the Bellman scale is still moving -- the controller
        # turns Bellman-loss noise into alpha noise. That is exactly where the
        # dynamics-varying environments diverge across seeds (hopper-friction
        # laco-auto gave 69.7 / 553.3 / 201.9 while the reward-varying ones were
        # stable). Warmup holds alpha until the scale settles, a longer EMA and a
        # per-step cap keep the correction from chasing that noise.
        # Defaults reproduce the original controller exactly.
        self.lc_auto_warmup   = int(kwargs.get('lc_auto_warmup', 0))
        self.lc_auto_ema      = float(kwargs.get('lc_auto_ema', 0.99))
        self.lc_auto_max_step = float(kwargs.get('lc_auto_max_step', float('inf')))
        self._log_alpha_lc = float(np.log(max(self.alpha_lc, 1e-8)))
        self._c_lc_ema = None

    # ------------------------------------------------------------------
    # LACO helpers (all inputs already detached from the encoder)
    # ------------------------------------------------------------------
    def _lc_proposals(self, z_tasks):
        """Build the proposal grid N_i for every task i in one tensor.

        z_tasks: [t, d] (detached, one latent per task).
        Returns [t, K, d] with K = (#enabled types) * (t - 1),
        or None when nothing can be built (t < 2 or all types off).
        """
        t, d = z_tasks.shape
        if t < 2:
            return None
        z_j = z_tasks.unsqueeze(0).expand(t, t, d)                    # [i, j, d]
        off_diag = ~torch.eye(t, dtype=torch.bool, device=z_tasks.device)
        z_others = z_j[off_diag].view(t, t - 1, d)                    # z^j, j != i
        z_i = z_tasks.unsqueeze(1)                                    # [t, 1, d]
        props = []
        if self.lc_use_cross:
            props.append(z_others)
        if self.lc_use_intp:
            lam = torch.rand(t, t - 1, 1, device=z_tasks.device)      # U(0,1), fresh per pair per step
            props.append(lam * z_i + (1. - lam) * z_others)
        if self.lc_use_extp:
            lam = 1. + self.eta * torch.rand(t, t - 1, 1, device=z_tasks.device)  # U(1, 1+eta)
            props.append(lam * z_i + (1. - lam) * z_others)
        if len(props) == 0:
            return None
        return torch.cat(props, dim=1)                                # [t, K, d]

    def _lc_penalty(self, t, b, obs, actions, z_tasks, q1_data, q2_data):
        """Latent-conservative gap, batched over the whole [t, b, K] grid.

        obs, actions: [t*b, .] flattened RL batch; z_tasks: [t, d] detached;
        q*_data: [t*b, 1] = Q(s, a, z^i) with detached z (matched term).
        Returns the raw mean gap (scalar tensor) or None.
        One critic forward per Q net over the full grid -- no python loop.
        """
        props = self._lc_proposals(z_tasks)
        if props is None:
            return None
        K = props.size(1)
        obs_g = obs.view(t, b, 1, -1).expand(-1, -1, K, -1)           # [t, b, K, obs]
        act_g = actions.view(t, b, 1, -1).expand(-1, -1, K, -1)       # [t, b, K, act]
        z_g = props.unsqueeze(1).expand(-1, b, -1, -1)                # [t, b, K, d]
        gaps = []
        for qf, q_data in ((self.qf1, q1_data), (self.qf2, q2_data)):
            q_grid = qf(t, b, obs_g, act_g, z_g).squeeze(-1)          # [t, b, K]
            # tau rescales the SHAPE of the soft-max only. Writing it as
            # tau*(logsumexp(q/tau) - logK) + logK keeps tau=1 bit-identical to
            # the original logsumexp(q) while tau->inf tends to mean(q)+logK;
            # without removing the tau*logK term the penalty would just grow
            # linearly in tau by a constant that carries no gradient.
            log_k = float(np.log(K))
            lse = self.lc_tau * (torch.logsumexp(q_grid / self.lc_tau, dim=-1)
                                 - log_k) + log_k
            gaps.append((lse - q_data.view(t, b)).mean())
        return 0.5 * (gaps[0] + gaps[1])

    def _alpha_lc_effective(self, loss_lc, qf_loss):
        """alpha for this step. Constant unless lc_auto, in which case a scalar
        dual step drives C_lc = alpha*gap/sqrt(qf_loss) toward lc_target.

        Both alpha*gap and sqrt(qf_loss) are in Q units, so C_lc is the
        dimensionally correct measure of how hard the penalty pushes relative
        to the Bellman term; holding it fixed is what makes one setting
        transfer across environments whose Q scale differs by ~100x.
        """
        if not self.lc_auto:
            return self.alpha_lc
        alpha = float(np.exp(self._log_alpha_lc))
        with torch.no_grad():
            denom = max(float(qf_loss.detach()) ** 0.5, 1e-6)
            c_lc = alpha * float(loss_lc.detach()) / denom
        e = self.lc_auto_ema
        self._c_lc_ema = c_lc if self._c_lc_ema is None \
            else e * self._c_lc_ema + (1.0 - e) * c_lc
        if self._n_train_steps_total >= self.lc_auto_warmup:
            step = self.lc_auto_lr * (self.lc_target - self._c_lc_ema)
            step = float(np.clip(step, -self.lc_auto_max_step, self.lc_auto_max_step))
            self._log_alpha_lc = float(np.clip(
                self._log_alpha_lc + step,
                np.log(self.lc_alpha_min), np.log(self.lc_alpha_max)))
        return alpha

    def _sample_latent_noise(self, z_flat, z_tasks):
        """delta ~ N(0, sigma^2 I), sigma = kappa * per-dim std of batch latents."""
        sigma = self.kappa * z_tasks.std(dim=0, unbiased=False, keepdim=True).detach()  # [1, d]
        delta = torch.randn_like(z_flat) * sigma                      # [t*b, d]
        delta_sq = delta.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-12)  # [t*b, 1]
        return delta, delta_sq

    def _critic_lip(self, t, b, obs, actions, z_flat, z_tasks, q1_data, q2_data):
        """Critic z-smoothness penalty + empirical Lipschitz estimate hat_Lz."""
        delta, delta_sq = self._sample_latent_noise(z_flat, z_tasks)
        z_pert = z_flat + delta
        lips, hats = [], []
        for qf, q_data in ((self.qf1, q1_data), (self.qf2, q2_data)):
            q_pert = qf(t, b, obs, actions, z_pert)                   # one extra forward
            diff = q_pert - q_data
            lips.append((diff.pow(2) / delta_sq).mean())
            hats.append((diff.abs() / delta_sq.sqrt()).mean().detach())
        return 0.5 * (lips[0] + lips[1]), 0.5 * (hats[0] + hats[1])

    # ------------------------------------------------------------------
    # Training step: verbatim CSROPrediction._take_step + marked LACO blocks
    # ------------------------------------------------------------------
    def _take_step(self, indices, context):
        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_in_context = context[:, :, obs_dim + action_dim].cpu().numpy()
        self.loss["non_sparse_ratio"] = len(reward_in_context[np.nonzero(reward_in_context)]) / np.size(reward_in_context)

        num_tasks = len(indices)
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        policy_outputs, task_z, task_z_vars= self.agent(obs, context, task_indices=indices)

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # Q and V networks
        # encoder will only get gradients from Q nets
        if self.allow_backward_z:
            q1_pred = self.qf1(t, b, obs, actions, task_z)
            q2_pred = self.qf2(t, b, obs, actions, task_z)
            v_pred = self.vf(t, b, obs, task_z.detach())
        else:
            q1_pred = self.qf1(t, b, obs, actions, task_z.detach())
            q2_pred = self.qf2(t, b, obs, actions, task_z.detach())
            v_pred = self.vf(t, b, obs, task_z.detach())

        c_loss = self._divergence.dual_critic_loss(obs, new_actions.detach(), actions, task_z.detach())
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()
        for _ in range(self._c_iter - 1):
            self._optimize_c(indices=indices, context=context)
        self.loss["c_loss"] = c_loss.item()

        div_estimate = self._divergence.dual_estimate(
            obs, new_actions, actions, task_z.detach())
        self.loss["div_estimate"] = torch.mean(div_estimate).item()

        with torch.no_grad():
            if self.use_brac and self.use_value_penalty:
                target_v_values = self.target_vf(t, b, next_obs, task_z) - self.get_alpha * div_estimate
            else:
                target_v_values = self.target_vf(t, b, next_obs, task_z)
        self.loss["target_v_values"] = torch.mean(target_v_values).item()

        if self.use_club:
            self.club_model_optimizer.zero_grad()
            z_target = self.agent.encode_no_mean(context).detach()
            z_param = self.club_model(context[...,:self.club_model.input_size])
            z_mean = z_param[..., :self.latent_dim]
            z_var = F.softplus(z_param[..., self.latent_dim:])
            club_model_loss = self.club_model_loss_weight * ((z_target - z_mean)**2/(2*z_var) + torch.log(torch.sqrt(z_var))).mean()
            club_model_loss.backward()
            self.loss["club_model_loss"] = club_model_loss.item()
            self.club_model_optimizer.step()

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()

        if self.use_club:
            z_target = self.agent.encode_no_mean(context)
            z_param = self.club_model(context[...,:self.club_model.input_size]).detach()
            z_mean = z_param[..., :self.latent_dim]
            z_var = F.softplus(z_param[..., self.latent_dim:])
            z_t, z_b, _ = z_mean.size()
            position = - ((z_target-z_mean)**2/z_var).mean()
            z_mean_expand = z_mean[:, :, None, :].expand(-1, -1, z_b, -1).reshape(z_t, z_b**2, -1)
            z_var_expand = z_var[:, :, None, :].expand(-1, -1, z_b, -1).reshape(z_t, z_b**2, -1)
            z_target_repeat = z_target.repeat(1, z_b, 1)
            negative = - ((z_target_repeat-z_mean_expand)**2/z_var_expand).mean()
            club_loss = self.club_loss_weight * (position - negative)
            club_loss.backward(retain_graph=True)
            self.loss["club_loss"] = club_loss.item()

        if self.use_FOCAL_cl:
            z_loss = self.z_loss_weight * self.FOCAL_z_loss(indices=indices, task_z=task_z, task_z_vars=task_z_vars, b=b)
            z_loss.backward(retain_graph=True)
            self.loss["z_loss"] = z_loss.item()

        if self.use_prediction:
            # extended_state = torch.cat([obs, actions, task_z], dim=-1)
            pred = self.decoder(t, b, obs, actions, task_z)
            target = torch.cat([(next_obs-obs)/self.std_state, (rewards_flat-self.mu_reward)/self.std_reward], dim=-1)
            prediction_loss = self.prediction_loss_weight * self.pred_criterion(pred, target)
            prediction_loss.backward(retain_graph=True)
            self.loss['prediction'] = prediction_loss.item()

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
                # backbone computed q*_pred on NON-detached z; the matched
                # term of the penalties must use detached z, so recompute.
                q1_data = self.qf1(t, b, obs, actions, z_flat_det)
                q2_data = self.qf2(t, b, obs, actions, z_flat_det)
            else:
                q1_data, q2_data = q1_pred, q2_pred                   # already detached-z forwards

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)

        # === LACO: critic regularizers added on top of the TD loss ===
        qf_total = qf_loss
        if self.use_lc:
            loss_lc = self._lc_penalty(t, b, obs, actions, z_tasks_det, q1_data, q2_data)
            if loss_lc is not None:
                alpha_eff = self._alpha_lc_effective(loss_lc, qf_loss)
                qf_total = qf_total + alpha_eff * loss_lc
        if self.use_lip:
            loss_lip, hat_Lz = self._critic_lip(t, b, obs, actions, z_flat_det, z_tasks_det, q1_data, q2_data)
            qf_total = qf_total + self.beta_lip * loss_lip

        qf_total.backward(retain_graph=True)
        self.loss["qf_loss"] = qf_loss.item()
        self.loss["q_target"] = torch.mean(q_target).item()
        self.loss["q1_pred"] = torch.mean(q1_pred).item()
        self.loss["q2_pred"] = torch.mean(q2_pred).item()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = torch.min(self.qf1(t, b, obs, new_actions, task_z.detach()),
                                        self.qf2(t, b, obs, new_actions, task_z.detach()))

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
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        # BRAC:
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
            pert_in = torch.cat([obs, z_flat_det + delta_pi], dim=1)
            # deterministic=True: mean/log_std heads only, no action sampling
            pert_out = self.agent.policy(t, b, pert_in, deterministic=True)
            mean_pert, log_std_pert = pert_out[1], pert_out[2]
            # analytic KL( N(mu1, s1) || N(mu2, s2) ) between pre-tanh Gaussians
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
        # optimize for c network (which computes dual-form divergences)
        # BRAC for training alpha:
        a_loss = -torch.mean(self._alpha_var * (div_estimate - self._target_divergence).detach())
        a_loss.backward()
        with torch.no_grad():
            self._alpha_var -= self.alpha_lr * self._alpha_var.grad
            # Manually zero the gradients after updating weights
            self._alpha_var.grad.zero_()
        self.loss["a_loss"] = a_loss.item()

        for i in range(len(self.agent.z_means[0])):
            z_mean = ptu.get_numpy(self.agent.z_means[0][i])
            name = f'train/z_mean_{i}'
            self.eval_statistics[name] = z_mean

        z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))

        self.eval_statistics['train/Z_variance'] = z_sig

        if self.use_club:
            self.eval_statistics['train/loss_club_model'] = ptu.get_numpy(club_model_loss)
            self.eval_statistics['train/loss_club'] = ptu.get_numpy(club_loss)
        if self.use_FOCAL_cl:
            self.eval_statistics['train/loss_focal'] = ptu.get_numpy(z_loss)
        if self.use_prediction:
            self.eval_statistics['train/loss_prediction'] = ptu.get_numpy(prediction_loss)

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
