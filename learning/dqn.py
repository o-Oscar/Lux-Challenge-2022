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

    def sample_actions(self, obs, masks):
        raise NotImplementedError

    def q_eval(self, obs, act, masks):
        raise NotImplementedError


@dataclasses.dataclass
class DqnConfig:
    agent: BaseAgent
    target_agent: BaseAgent
    env: Env
    save_path: Path
    device: th.device
    wandb: bool = False
    gamma: float = 0.99
    epoch_per_save: int = 0
    update_nb: int = 1000


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


def actions_to_env(network_actions):
    actions = network_actions.detach().cpu().numpy()
    to_return = {team: actions[i] for i, team in enumerate(teams)}
    return to_return


def multi_agent_rollout(env: Env, agent: BaseAgent, device, max_ep_len=5):
    obs, masks, unit_pos = env.reset(seed=42)

    all_obs = []
    all_rewards = []
    all_masks = []
    all_actions = []
    all_unit_pos = []

    for t in range(max_ep_len):
        network_obs = obs_to_network(obs, device)

        network_masks = mask_to_network(masks, device)
        actions = agent.sample_actions(network_obs, network_masks)
        actions = th.tensor(
            np.random.randint(0, 5, size=actions.shape), dtype=th.long, device=device
        )
        actions = actions_to_env(actions)

        new_obs, rewards, new_mask, done, new_unit_pos = env.step(actions)

        all_obs.append(obs)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_masks.append(masks)
        all_unit_pos.append(unit_pos)

        obs = new_obs
        unit_pos = new_unit_pos
        masks = new_mask

        if done:
            break

    env.save(full_save=False, convenient_save=True)

    return {
        "obs": all_obs,
        "actions": all_actions,
        "rewards": all_rewards,
        "masks": all_masks,
        "unit_pos": all_unit_pos,
    }


def map_robots_grid(robot_pos, next_robot_pos, new_value):
    to_return = new_value * 0
    for robot_id, next_pos in next_robot_pos.items():
        if robot_id in robot_pos:
            cur_pos = robot_pos[robot_id]
            to_return[cur_pos[1], cur_pos[0]] = new_value[next_pos[1], next_pos[0]]
    return to_return


def map_robots_grids(all_robot_pos, all_next_robot_pos, new_value):
    to_return = th.zeros_like(new_value)
    for i, (robot_pos, next_robot_pos) in enumerate(
        zip(all_robot_pos, all_next_robot_pos)
    ):
        for robot_id, next_pos in next_robot_pos.items():
            if robot_id in robot_pos:
                cur_pos = robot_pos[robot_id]
                to_return[i, cur_pos[1], cur_pos[0]] = new_value[
                    i, next_pos[1], next_pos[0]
                ]
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

        self.all_obs = []
        self.all_actions = []
        self.all_rewards = []
        self.all_masks = []
        self.all_unit_pos = []

    def fill(self, batch_size: int):

        cur_size = len(self)

        while len(self) < batch_size + cur_size:
            self.expand(multi_agent_rollout(self.env, self.agent, self.device))

    def expand(self, rollout):
        for team in teams:
            self.all_obs.append(get_team_rollout(rollout, "obs", team))
            self.all_actions.append(get_team_rollout(rollout, "actions", team))
            self.all_rewards.append(get_team_rollout(rollout, "rewards", team))
            self.all_masks.append(get_team_rollout(rollout, "masks", team))
            self.all_unit_pos.append(get_team_rollout(rollout, "unit_pos", team))

    def calc_q_target(self, agent: BaseAgent):
        with th.no_grad():
            to_save = []
            for obs, masks, rewards, done in self.sample_targets(10):
                all_q = []
                for i in range(10):
                    act = agent.sample_actions(obs, masks)
                    all_q.append(agent.q_eval(obs, act, masks).detach().cpu().numpy())
                mean_q = np.mean(np.stack(all_q), axis=0)
                to_save.append(mean_q)
            to_save = np.concatenate(to_save, axis=0)

            self.all_q_target = []
            count = 0
            for i in range(len(self.all_actions)):
                cur_q_targets = []
                for t in range(len(self.all_actions[i])):
                    q_target = self.all_rewards[i][t]
                    if len(self.all_unit_pos[i]) < t - 1:
                        q_target = q_target + map_robots_grids(
                            self.all_unit_pos[i][t],
                            self.all_unit_pos[i][t + 1],
                            mean_q[count + 1],
                        )
                    cur_q_targets.append(q_target)
                    count += 1
                self.all_q_target.append(cur_q_targets)

    def sample_targets(self, batch_size):
        ids = []
        for i in range(len(self.all_actions)):
            ids = ids + [(i, t) for t in range(len(self.all_actions[i]))]

        # ids = np.random.permutation(ids)
        for batch_start in range(0, len(ids), batch_size):
            batch_end = batch_start + batch_size
            if batch_end <= len(ids):
                yield self.generate_target_batch(ids[batch_start:batch_end])

    def generate_target_batch(self, ids):
        obs = []
        masks = []
        rewards = []
        done = []

        for id in ids:
            obs.append(self.all_obs[id[0]][id[1]])
            masks.append(self.all_masks[id[0]][id[1]])

            rewards.append(self.all_rewards[id[0]][id[1]])
            done.append(1 if len(self.all_obs[id[0]]) == id[1] - 1 else 0)

        obs = np.stack(obs, axis=0)
        masks = np.stack(masks, axis=0)

        obs = th.tensor(obs, device=self.device, dtype=th.float32)
        masks = th.tensor(masks, device=self.device, dtype=th.float32)

        rewards = th.tensor(np.array(rewards), device=self.device, dtype=th.float32)
        done = th.tensor(done, device=self.device, dtype=th.float32)

        return (
            obs,
            masks,
            rewards,
            done,
        )

    def __len__(self):
        return sum([len(x) for x in self.all_actions])

    def sample(self, batch_size):
        ids = []
        for i in range(len(self.all_actions)):
            ids = ids + [(i, t - 1) for t in range(len(self.all_actions[i]))]

        ids = np.random.permutation(ids)
        for batch_start in range(0, len(ids), batch_size):
            batch_end = batch_start + batch_size
            if batch_end <= len(ids):
                yield self.generate_batch(ids[batch_start:batch_end])

    def generate_batch(self, ids):
        obs = []
        robot_pos = []
        masks = []
        actions = []
        new_obs = []
        new_robot_pos = []
        new_masks = []
        rewards = []
        done = []
        targets = []

        for id in ids:
            obs.append(self.all_obs[id[0]][id[1]])
            robot_pos.append(self.all_unit_pos[id[0]][id[1]])
            masks.append(self.all_masks[id[0]][id[1]])
            actions.append(self.all_actions[id[0]][id[1]])

            new_obs.append(self.all_obs[id[0]][id[1] + 1])
            new_robot_pos.append(self.all_unit_pos[id[0]][id[1] + 1])
            new_masks.append(self.all_masks[id[0]][id[1] + 1])

            rewards.append(self.all_rewards[id[0]][id[1]])
            done.append(1 if id[1] == -1 else 0)

            targets.append(self.all_q_target[id[0]][id[1]])

        obs = np.stack(obs, axis=0)
        actions = np.stack(actions, axis=0)
        new_obs = np.stack(new_obs, axis=0)
        masks = np.stack(masks, axis=0)
        new_masks = np.stack(new_masks, axis=0)
        targets = np.stack(targets, axis=0)

        obs = th.tensor(obs, device=self.device, dtype=th.float32)
        actions = th.tensor(actions, device=self.device, dtype=th.float32)
        new_obs = th.tensor(new_obs, device=self.device, dtype=th.float32)
        masks = th.tensor(masks, device=self.device, dtype=th.float32)
        new_masks = th.tensor(new_masks, device=self.device, dtype=th.float32)

        rewards = th.tensor(np.array(rewards), device=self.device, dtype=th.float32)
        done = th.tensor(done, device=self.device, dtype=th.float32)

        targets = th.tensor(targets, device=self.device, dtype=th.float32)

        return (
            obs,
            robot_pos,
            masks,
            actions,
            new_obs,
            new_robot_pos,
            new_masks,
            rewards,
            done,
            targets,
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


def start_dqn(config: DqnConfig):

    # if config.epoch_per_save > 0 and config.save_path.is_dir():
    #     raise NameError("Save folder already exists. Default is to not override")
    # if config.epoch_per_save > 0:
    #     config.save_path.mkdir(exist_ok=False, parents=True)

    if config.wandb:
        wandb.init(project="lux_ai_ppo", name=config.save_path.name)

    env = config.env

    device = config.device

    agent = config.agent.to(device)
    target_agent = config.target_agent.to(device)
    target_agent.load_state_dict(agent.state_dict())

    optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

    buffer = ReplayBuffer(env, agent, device)

    start_time = time.time()

    q_loss = MeanLogger()
    mean_ratio = MeanLogger()
    mean_reward = MeanLogger()

    buffer.fill(256)
    # buffer.fill(1024)
    buffer.calc_q_target(target_agent)

    for update in range(config.update_nb):

        start_time = time.time()
        print("Update {} | Trajs computation ".format(update), end="", flush=True)

        if config.epoch_per_save > 0 and (update + 1) % config.epoch_per_save == 0:
            model_name = "{:04d}".format(update * 0)
            th.save(agent.state_dict(), config.save_path / model_name)

        q_loss.reset()

        # exit()

        print("| Learning ", end="", flush=True)

        for epoch in range(1):
            for (
                obs,
                robot_pos,
                masks,
                actions,
                new_obs,
                new_robot_pos,
                new_masks,
                rewards,
                done,
                q_target,
            ) in buffer.sample(batch_size=32):

                # with th.no_grad():
                #     new_actions = target_agent.sample_actions(new_obs, new_masks)
                #     new_q = target_agent.q_eval(new_obs, new_actions, new_masks)
                #     done = done.view((-1, 1, 1))
                #     q_target = config.gamma * new_q * (1 - done)
                #     q_target = q_target * th.max(new_masks, dim=1).values
                #     q_target = rewards + map_robots_grids(
                #         robot_pos, new_robot_pos, q_target
                #     )

                q_pred = agent.q_eval(obs, actions, masks)
                loss = m_mean(th.square(q_pred - q_target), th.max(masks, dim=1).values)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                q_loss.update(loss.item())

        to_log = {}

        # mean reward computation
        mean_rew = 0
        for rew, mask in zip(buffer.all_rewards, buffer.all_masks):
            m = np.array(mask)
            m = np.max(m, axis=1)
            r = np.array(rew)
            mean_rew += np.sum(m * r) / np.sum(m) / len(teams)

        # logging
        to_log["main/q_loss"] = q_loss.value

        # if q_loss.value < 1e-5:
        #     buffer.fill(70)
        #     buffer.calc_q_target(target_agent)

        # to_log["main/value_loss"] = value_loss.value

        # to_log["infos/policy_loss"] = policy_loss.value
        # to_log["infos/policy_loss"] = policy_loss.value
        # to_log["infos/entropy"] = entropy_logger.value
        # to_log["infos/approx_kl"] = approx_kl_logger.value
        # to_log["infos/clipfrac"] = clipfrac_logger.value
        # to_log["infos/explained_variance"] = explained_variance.value
        # to_log["infos/mean_ratio"] = mean_ratio.value

        if config.wandb:
            wandb.log(to_log)

        print("| Done ({:.03f}s)".format(time.time() - start_time))

        if not config.wandb:
            print(to_log)
