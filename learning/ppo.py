import argparse
import dataclasses
import itertools
import os
import random
import time
from distutils.util import strtobool
from pathlib import Path

import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import wandb
from utils import teams
from utils.env import Env, get_env


class BaseAgent(nn.Module):
    def __init__(self):
        super().__init__()

    def get_value(self, obs):
        raise NotImplementedError

    def get_action(self, obs, masks, action=None):
        raise NotImplementedError


@dataclasses.dataclass
class PPOConfig:
    agent: BaseAgent
    env: Env
    save_path: Path
    device: th.device
    wandb: bool = False
    epoch_per_save: int = 0
    min_batch_size: int = 32
    update_nb: int = 1000


# TODO : add that to the ppo config
clip_coef = 0.1
norm_adv = True
ent_coef = 0
vf_coef = 1
max_grad_norm = 0.5


def obs_to_network(obs, device):
    to_return = []
    for team in teams:
        to_return.append(obs[team])

    to_return = np.stack(to_return)
    to_return = th.tensor(to_return, device=device, dtype=th.float32)

    return to_return


def mask_to_network(masks, device):
    to_return = []
    for team in teams:
        to_return.append(masks[team])

    to_return = np.stack(to_return)
    to_return = th.tensor(to_return, device=device, dtype=th.float32)

    return to_return


def actions_to_env(network_actions, obs):
    to_return = {team: network_actions[i] for i, team in enumerate(teams)}
    return to_return


def multi_agent_rollout(env: Env, agent: BaseAgent, device, max_ep_len=1100):
    obs, masks, unit_pos = env.reset()

    all_obs = [obs]
    all_rewards = []
    all_masks = [masks]
    all_actions = []
    all_log_prob = []
    all_values = []
    all_unit_pos = [unit_pos]

    no_unit = False

    for t in range(max_ep_len):
        network_obs = obs_to_network(obs, device)

        # if t == 0:
        #     # print(th.sum(th.tensor(obs[teams[0]], device=device, dtype=th.float32)))
        #     print(th.where(th.tensor(masks[teams[0]], device=device, dtype=th.float32)))

        network_masks = mask_to_network(masks, device)
        actions, log_prob, _, value = agent.get_action(network_obs, network_masks)
        actions = actions_to_env(actions, obs)
        log_prob = actions_to_env(log_prob, obs)
        value = actions_to_env(value, obs)

        # if network_obs["grid"].shape[0] > 0:
        #
        # else:
        #     actions = {team: {} for team in teams}
        #     log_prob = {team: {} for team in teams}
        #     value = {team: {} for team in teams}

        obs, rewards, masks, done, unit_pos = env.step(actions)
        # print(np.min(rewards[teams[0]]), np.max(rewards[teams[0]]))

        all_values.append(value)

        if done:
            break

        all_obs.append(obs)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_masks.append(masks)
        all_log_prob.append(log_prob)
        all_unit_pos.append(unit_pos)

    env.save(full_save=False, convenient_save=True)
    # exit()
    # if no_unit:
    #     print("no units !!")
    #     exit()

    return {
        "obs": all_obs,
        "actions": all_actions,
        "rewards": all_rewards,
        "masks": all_masks,
        "logprob": all_log_prob,
        "values": all_values,
        "unit_pos": all_unit_pos,
    }


def map_robots_grid(robot_pos, next_robot_pos, new_value):
    to_return = new_value * 0
    for robot_id, next_pos in next_robot_pos.items():
        if robot_id in robot_pos:
            cur_pos = robot_pos[robot_id]
            to_return[cur_pos[1], cur_pos[0]] = new_value[next_pos[1], next_pos[0]]
    return to_return


def get_team_rollout(
    rollout,
    key,
    team,
):
    to_return = [x[team] for x in rollout[key]]
    return to_return


class ReplayBuffer:
    def __init__(self, env: Env, agent: BaseAgent, device):
        self.env = env
        self.agent = agent
        self.device = device

        self.gamma = 0.99
        self.lamb = 0.95

    def fill(self, batch_size: int):
        self.all_obs = []
        self.all_actions = []
        self.all_rewards = []
        self.all_masks = []
        self.all_logprob = []
        self.all_values = []
        self.all_unit_pos = []

        while len(self) < batch_size:
            self.expand(multi_agent_rollout(self.env, self.agent, self.device))

        self.compute_advantage()
        self.compute_gae()

    def expand(self, rollout):
        for team in teams:
            self.all_obs.append(get_team_rollout(rollout, "obs", team))
            self.all_actions.append(get_team_rollout(rollout, "actions", team))
            self.all_rewards.append(get_team_rollout(rollout, "rewards", team))
            self.all_masks.append(get_team_rollout(rollout, "masks", team))
            self.all_logprob.append(get_team_rollout(rollout, "logprob", team))
            self.all_values.append(get_team_rollout(rollout, "values", team))
            self.all_unit_pos.append(get_team_rollout(rollout, "unit_pos", team))

    def compute_advantage(self):
        self.all_advantages = []
        for game_id in range(len(self.all_rewards)):
            game_advantages = []
            for t in range(len(self.all_rewards[game_id])):
                rew = th.tensor(self.all_rewards[game_id][t], device=self.device)
                prec_value = self.all_values[game_id][t]
                new_value = prec_value * 0
                if t + 1 < len(self.all_unit_pos[game_id]):
                    robot_pos = self.all_unit_pos[game_id][t]
                    next_robot_pos = self.all_unit_pos[game_id][t + 1]
                    new_value = self.all_values[game_id][t + 1]
                    new_value = map_robots_grid(robot_pos, next_robot_pos, new_value)

                advantage = rew + self.gamma * new_value - prec_value
                game_advantages.append(advantage)
            self.all_advantages.append(game_advantages)

    def compute_gae(self):
        self.all_gae = []
        for game_id in range(len(self.all_advantages)):
            gae = self.all_advantages[game_id][-1]
            # gae = gae * 0
            cur_gae = [gae]
            for t in reversed(range(len(self.all_advantages[game_id]) - 1)):
                robot_pos = self.all_unit_pos[game_id][t]
                next_robot_pos = self.all_unit_pos[game_id][t + 1]
                gae = map_robots_grid(robot_pos, next_robot_pos, gae)
                gae = self.all_advantages[game_id][t] + self.gamma * self.lamb * gae
                # gae = gae * 0
                cur_gae.append(gae)

            self.all_gae.append(list(reversed(cur_gae)))

        # game_id = 0
        # r_id = list(self.all_unit_pos[game_id][0].keys())[0]
        # print(r_id)
        # for t in range(len(self.all_unit_pos[game_id])):
        #     robot_pos = self.all_unit_pos[game_id][t]
        #     for robot_id, cur_pos in robot_pos.items():
        #         if robot_id == r_id:

        #             gae = self.all_gae[game_id][t]
        #             value = self.all_values[game_id][t]
        #             reward = self.all_rewards[game_id][t]
        #             advantage = self.all_advantages[game_id][t]

        #             print(
        #                 t,
        #                 gae[cur_pos[1], cur_pos[0]].item(),
        #                 advantage[cur_pos[1], cur_pos[0]].item(),
        #                 value[cur_pos[1], cur_pos[0]].item(),
        #                 reward[cur_pos[1], cur_pos[0]].item(),
        #                 (cur_pos[1], cur_pos[0]),
        #             )
        # exit()

        # print(
        #     len(self.all_obs[game_id]),
        #     len(self.all_actions[game_id]),
        #     len(self.all_rewards[game_id]),
        #     len(self.all_masks[game_id]),
        #     len(self.all_logprob[game_id]),
        #     len(self.all_values[game_id]),
        #     len(self.all_unit_pos[game_id]),
        #     len(self.all_advantages[game_id]),
        #     len(self.all_gae[game_id]),
        # )
        # exit()

    def __len__(self):
        return sum([len(x) for x in self.all_actions])

    def sample(self, batch_size):
        ids = []
        for i in range(len(self.all_actions)):
            ids = ids + [(i, t) for t in range(len(self.all_actions[i]))]

        ids = np.random.permutation(ids)
        for batch_start in range(0, len(ids), batch_size):
            batch_end = batch_start + batch_size
            if batch_end <= len(ids):
                yield self.generate_batch(ids[batch_start:batch_end])

    def generate_batch(self, ids):
        ret_obs = []
        ret_actions = []
        ret_logprob = []
        ret_gae = []
        ret_masks = []
        ret_returns = []

        for id in ids:
            ret_obs.append(self.all_obs[id[0]][id[1]])
            ret_actions.append(self.all_actions[id[0]][id[1]])
            ret_logprob.append(self.all_logprob[id[0]][id[1]])
            ret_gae.append(self.all_gae[id[0]][id[1]])
            ret_masks.append(self.all_masks[id[0]][id[1]])
            ret_returns.append(
                self.all_gae[id[0]][id[1]] + self.all_values[id[0]][id[1]]
            )

        # print(ret_actions)
        # print(ret_gae[0])

        ret_obs = np.stack(ret_obs)
        ret_masks = np.stack(ret_masks)

        ret_obs = th.tensor(ret_obs, device=self.device, dtype=th.float32)
        ret_actions = th.stack(ret_actions).detach()
        ret_logprob = th.stack(ret_logprob).detach()
        ret_gae = th.stack(ret_gae).detach()
        ret_masks = th.tensor(ret_masks, device=self.device, dtype=th.float32)
        ret_returns = th.stack(ret_returns).detach()

        # print("shapes")
        # print(ret_obs.shape)
        # print(ret_actions.shape)
        # print(ret_logprob.shape)
        # print(ret_gae.shape)
        # print(ret_masks.shape)
        # print(ret_returns.shape)
        # exit()

        return (
            ret_obs,
            ret_actions,
            ret_logprob,
            ret_gae,
            ret_masks,
            ret_returns,
        )


class MeanLogger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.sum = 0

    def update(self, value, n=1):
        self.n += n
        self.sum += value * n

    @property
    def value(self):
        return self.sum / self.n


def m_mean(array, mask):
    return (array * mask).sum() / (mask.sum() + 1e-5)


def m_var(array, mask):
    m = m_mean(array, mask)
    return m_mean((array - m) ** 2, mask)


def start_ppo(config: PPOConfig):

    if config.epoch_per_save > 0 and config.save_path.is_dir():
        raise NameError("Save folder already exists. Default is to not override")
    if config.epoch_per_save > 0:
        config.save_path.mkdir(exist_ok=False, parents=True)

    if config.wandb:
        wandb.init(project="lux_ai_ppo", name=config.save_path.name)

    env = config.env

    device = config.device

    agent = config.agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

    buffer = ReplayBuffer(env, agent, device)

    start_time = time.time()

    value_loss = MeanLogger()
    policy_loss = MeanLogger()
    entropy_logger = MeanLogger()
    old_approx_kl_logger = MeanLogger()
    approx_kl_logger = MeanLogger()
    clipfrac_logger = MeanLogger()
    explained_variance = MeanLogger()
    mean_ratio = MeanLogger()
    mean_reward = MeanLogger()

    for update in range(config.update_nb):

        start_time = time.time()
        print("Update {} | Trajs computation ".format(update), end="", flush=True)

        if config.epoch_per_save > 0 and (update + 1) % config.epoch_per_save == 0:
            model_name = "{:04d}".format(update)
            th.save(agent.state_dict(), config.save_path / model_name)

        value_loss.reset()
        policy_loss.reset()
        entropy_logger.reset()
        old_approx_kl_logger.reset()
        approx_kl_logger.reset()
        clipfrac_logger.reset()
        explained_variance.reset()
        mean_ratio.reset()
        mean_reward.reset()

        buffer.fill(config.min_batch_size)
        # exit()

        print("| Learning ", end="", flush=True)

        for epoch in range(4):
            for obs, act, logprob, gae, masks, rets in buffer.sample(
                batch_size=len(buffer) // 4
            ):

                _, newlogprob, entropy, newvalue = agent.get_action(obs, masks, act)
                logratio = newlogprob - logprob
                ratio = logratio.exp()

                robot_mask = th.max(masks, dim=1).values

                ratio = ratio * robot_mask

                with th.no_grad():
                    old_approx_kl = (-logratio * robot_mask).sum() / robot_mask.sum()
                    approx_kl = m_mean((ratio - 1) - logratio, robot_mask)
                    clipfracs = m_mean(
                        ((ratio - 1.0).abs() > clip_coef).float(), robot_mask
                    )

                if norm_adv:
                    gae_mean = m_mean(gae, robot_mask)
                    gae_std = th.sqrt(m_var(gae, robot_mask))
                    gae = (gae - gae_mean) / (gae_std + 1e-8)

                # Policy loss
                pg_loss1 = -gae * ratio
                pg_loss2 = -gae * th.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = m_mean(th.max(pg_loss1, pg_loss2), robot_mask)

                # Value loss
                v_loss = 0.5 * m_mean((newvalue - rets) ** 2, robot_mask)

                entropy_loss = m_mean(entropy, robot_mask)
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

                n = robot_mask.sum().item()
                value_loss.update(v_loss.detach().cpu().item(), n)
                policy_loss.update(pg_loss.detach().cpu().item(), n)
                entropy_logger.update(entropy_loss.detach().cpu().item(), n)
                old_approx_kl_logger.update(
                    th.mean(old_approx_kl).detach().cpu().item(), n
                )
                approx_kl_logger.update(
                    m_mean(approx_kl, robot_mask).detach().cpu().item(), n
                )
                clipfrac_logger.update(clipfracs.item(), n)

                y_pred, y_true = newvalue, rets
                var_y = m_var(y_true, robot_mask).item()
                explained_var = (
                    np.nan
                    if var_y == 0
                    else (1 - m_var(y_true - y_pred, robot_mask).item() / var_y)
                )
                explained_variance.update(explained_var, n)
                mean_ratio.update(m_mean(ratio, robot_mask), n)

        to_log = {}

        # mean reward computation
        mean_rew = 0
        for rew, mask in zip(buffer.all_rewards, buffer.all_masks):
            m = np.array(mask)
            m = np.max(m, axis=1)[:-1]
            r = np.array(rew)
            mean_rew += np.sum(m * r) / np.sum(m) / len(teams)

        # logging
        to_log["main/mean_reward"] = mean_rew
        to_log["main/value_loss"] = value_loss.value

        to_log["infos/policy_loss"] = policy_loss.value
        to_log["infos/policy_loss"] = policy_loss.value
        to_log["infos/entropy"] = entropy_logger.value
        to_log["infos/approx_kl"] = approx_kl_logger.value
        to_log["infos/clipfrac"] = clipfrac_logger.value
        to_log["infos/explained_variance"] = explained_variance.value
        to_log["infos/mean_ratio"] = mean_ratio.value

        if config.wandb:
            wandb.log(to_log)

        print("| Done ({:.03f}s)".format(time.time() - start_time))

        if not config.wandb:
            print(to_log)
