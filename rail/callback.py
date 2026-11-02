"""Original RAIL update and evaluation logic with local logging/checkpoints."""

from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization
from rail.models import InfoNCELoss
from rail.logging import DictLogger


class RAILCallback(BaseCallback):
    def __init__(self, eval_env, config, output, eval_episodes=60):
        super().__init__()
        self.config = config
        self.output = Path(output)
        self.eval_episodes = eval_episodes
        self.iteration = 0
        self.dict_logger = DictLogger()
        self.eval_env = eval_env
        self.eval_interval = self.config["eval_interval"]
        self.high_student_eval = False
        self.high_action_alignment = False
        self.enough_cl_training = False
        self.perfect_student = False
        self.max_timesteps = self.config["max_steps"]
        self.method_name = self.config["method_name"]
        self.method_cfg = self.config["methods"][self.method_name]

    def save_model(self, file_name):
        path = self.output / "checkpoints" / Path(file_name).stem
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        if isinstance(self.model.env, VecNormalize):
            self.model.env.save(path.with_suffix(".pkl"))

    def log(self, metrics):
        with (self.output / "metrics.jsonl").open("a") as file:
            file.write(json.dumps({"timesteps": self.num_timesteps, **metrics}) + "\n")

    def _on_training_start(self):
        extractor = self.model.policy.features_extractor
        self.contrastive_fn = InfoNCELoss().to(self.model.device)
        self.contrastive_optim = torch.optim.Adam(
            list(extractor.parameters()) + list(self.contrastive_fn.parameters()),
            lr=0.001,
        )

    def _on_step(self) -> None:
        if self.perfect_student:
            self.save_model("converged.pt")
            return False
        return True

    def _on_rollout_end(self) -> None:
        self.iteration += 1
        if self.iteration % self.eval_interval == 0:
            self.eval_policy(self.iteration, eval_student=False)
            SR = self.eval_policy(self.iteration, eval_student=True)
            print(f"saving the model")
            self.save_model(f"{self.num_timesteps}_{SR:.2f}.pt")
        if self.config["align_student"]:
            self.align_policy()
        return True

    @torch.no_grad()
    def _actions_from_features(self, features):
        actions = (
            self.model.policy.get_actions_from_features(
                deterministic=True, features=features
            )
            .cpu()
            .numpy()
        )
        if isinstance(self.model.action_space, spaces.Box):
            actions = np.clip(
                actions, self.model.action_space.low, self.model.action_space.high
            )
        return actions

    @torch.no_grad()
    def eval_policy(self, iteration, eval_student=False):
        policy_name = "teacher" if not eval_student else "student"
        print(f"Evaluating {policy_name} policy at iteration {iteration}", end=" ")
        model = self.model
        extractor = model.policy.features_extractor
        if self.config["normalize_env"]:
            sync_envs_normalization(self.model.env, self.eval_env)
            self.eval_env.training = False
            self.eval_env.norm_reward = False
        n_envs = self.eval_env.num_envs
        n_eps_per_env = max(1, self.eval_episodes // n_envs)
        done_buffer = np.zeros(n_envs, dtype=float)
        current_return = np.zeros(n_envs, dtype=float)
        self.dict_logger.clean()
        obs_dict = self.eval_env.reset()
        while np.any(done_buffer < n_eps_per_env):
            obs_tensor = obs_as_tensor(obs_dict, device=model.device)
            s_s = extractor.forward_student(obs_tensor)
            s_t = extractor.forward_teacher(obs_tensor)
            student_actions = self._actions_from_features(extractor._add_noise(s_s))
            teacher_actions = self._actions_from_features(extractor._add_noise(s_t))
            actions = student_actions if eval_student else teacher_actions
            obs_dict, reward, dones, infos = self.eval_env.step(actions)
            current_return += reward
            for idx in np.where(dones & (done_buffer < n_eps_per_env))[0]:
                self.dict_logger.add("success", infos[idx]["is_success"])
                self.dict_logger.add("return", current_return[idx])
            done_buffer += dones
            current_return[dones] = 0.0
            action_alignment = (teacher_actions == student_actions).mean()
            cos_similarity = F.cosine_similarity(s_s, s_t, dim=-1).mean()
            self.dict_logger.add("action_alignment", action_alignment.item())
            self.dict_logger.add("cosine_similarity", cos_similarity.item())
        self.eval_env.reset()
        logs = self.dict_logger.dump()
        success_mean = logs["success"]
        print(
            f" SR: {logs['success']:.2f}, R: {logs['return']:.2f}, AA: {logs['action_alignment']:.2f}, CS: {logs['cosine_similarity']:.2f}"
        )
        if eval_student:
            self.high_student_eval = logs["success"] >= 0.98
            self.enough_cl_training = self.high_student_eval
            print("Enough CL training: ", self.enough_cl_training)
        logs = self.dict_logger.dump(prefix=f"eval/{policy_name}_")
        self.log(logs)
        if (
            eval_student
            and logs["eval/student_success"] >= 0.995
            and (not self.perfect_student)
        ):
            self.perfect_student = True
        self.dict_logger.clean()
        return success_mean

    def align_policy(self):
        model = self.model
        extractor = model.policy.features_extractor
        extractor.train()
        batch_size = self.method_cfg["batch_size"]
        self.dict_logger.clean()
        obs_all = next(model.rollout_buffer.get(batch_size=None)).observations
        n_samples = len(next(iter(obs_all.values())))
        use_stability = True
        stability_targets = None
        if use_stability:
            old_extractor = deepcopy(extractor).eval()
            with torch.no_grad():
                s_t_old = old_extractor.forward_teacher(obs_all)
                s_s_old = old_extractor.forward_student(obs_all)
                latent_t_old = model.policy.mlp_extractor.policy_net(s_t_old)
                latent_s_old = model.policy.mlp_extractor.policy_net(s_s_old)
                stability_targets = {
                    "logits_t": model.policy.action_net(latent_t_old),
                    "logits_s": model.policy.action_net(latent_s_old),
                    "values_t": model.policy.value_net(latent_t_old),
                    "values_s": model.policy.value_net(latent_s_old),
                }
            del old_extractor, s_t_old, s_s_old, latent_t_old, latent_s_old
        for epoch in range(self.method_cfg["n_align_epochs"]):
            permutation = torch.randperm(n_samples, device=model.device)
            for start in range(0, n_samples, batch_size):
                indices = permutation[start : start + batch_size]
                obs_dict = {key: value[indices] for key, value in obs_all.items()}
                s_t = extractor.forward_teacher(obs_dict)
                s_s = extractor.forward_student(obs_dict, augment=True)
                loss_stability = None
                if use_stability:
                    latent_pi_teacher = model.policy.mlp_extractor.policy_net(s_t)
                    latent_pi_student = model.policy.mlp_extractor.policy_net(s_s)
                    logits_teacher = model.policy.action_net(latent_pi_teacher)
                    logits_student = model.policy.action_net(latent_pi_student)
                    values_teacher = model.policy.value_net(latent_pi_teacher)
                    values_student = model.policy.value_net(latent_pi_student)
                    value_stability = F.mse_loss(
                        values_teacher, stability_targets["values_t"][indices]
                    ) + F.mse_loss(
                        values_student, stability_targets["values_s"][indices]
                    )
                    logit_stability = F.mse_loss(
                        logits_teacher, stability_targets["logits_t"][indices]
                    ) + F.mse_loss(
                        logits_student, stability_targets["logits_s"][indices]
                    )
                    loss_stability = (value_stability + logit_stability) / 2
                cos_similarity = F.cosine_similarity(s_t, s_s, dim=-1)
                loss_alignment = 1 - cos_similarity.mean()
                if self.method_cfg["discrete_obs"]:
                    loss_cont, acc_cont = self.contrastive_fn.get_loss_with_duplicates(
                        obs_dict["teacher_obs"], obs_dict["student_obs"], s_t, s_s
                    )
                else:
                    loss_cont, acc_cont = self.contrastive_fn(s_t, s_s)
                loss = loss_cont
                if use_stability:
                    loss += loss_stability
                loss += loss_alignment
                if self.config["align_student"] and (not self.enough_cl_training):
                    self.contrastive_optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(extractor.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(
                        self.contrastive_fn.parameters(), max_norm=1.0
                    )
                    self.contrastive_optim.step()
                with torch.no_grad():
                    teacher_actions = model.policy.get_actions_from_features(
                        deterministic=True, features=s_t
                    )
                    student_actions = model.policy.get_actions_from_features(
                        deterministic=True, features=s_s
                    )
                    action_alignment = torch.mean(
                        (teacher_actions == student_actions).float()
                    )
                    l1_loss = F.l1_loss(s_t, s_s).cpu()
                if epoch == 0:
                    self.dict_logger.add("feature_alignment", l1_loss.item())
                    self.dict_logger.add(
                        "cosine_similarity", cos_similarity.mean().item()
                    )
                    self.dict_logger.add("action_alignment", action_alignment.item())
                    self.dict_logger.add("contrastive_loss", loss_cont.item())
                    self.dict_logger.add("contrastive_accuracy", acc_cont.item())
        self.dict_logger.add("scale", self.contrastive_fn.scale.item())
        log_dict = self.dict_logger.dump(prefix="contrastive/")
        self.dict_logger.clean()
        self.log(log_dict)
