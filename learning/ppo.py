import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from utils.env import get_env, Env
from utils import teams
import itertools
import wandb
from pathlib import Path


# args.batch_size = int(args.num_envs * args.num_steps)
# args.minibatch_size = int(args.batch_size // args.num_minibatches)

BIG_NUMBER = 1e10


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env: Env):
        super().__init__()
        in_channels = env.obs_generator.channel_nb
        self.critic = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding="same"),
        )
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, env.action_handler.action_nb, 3, padding="same"),
        )

    def get_value(self, obs):
        return self.critic(obs).squeeze(dim=1)

    def get_action(self, obs, masks, action=None):
        logits = self.actor(obs)
        logits = logits - BIG_NUMBER * (1 - masks)
        logits = logits.permute(0, 2, 3, 1)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(obs).squeeze(dim=1),
        )

    # def get_greedy_action(self, obs, masks, action=None):
    #     logits = self.actor(self.act_backbone(obs))
    #     logits = logits - BIG_NUMBER * (1 - masks)
    #     probs = Categorical(logits=logits)
    #     if action is None:
    #         action = th.argmax(logits, dim=-1)
    #     return (
    #         action,
    #         probs.log_prob(action),
    #         probs.entropy(),
    #         self.critic(self.critic_backbone(obs)).view(-1),
    #     )


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


def multi_agent_rollout(env: Env, agent: Agent, device, max_ep_len=1100):
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
        print(t)
        network_obs = obs_to_network(obs, device)

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

        obs, rewards, masks, done, units_pos = env.step(actions)

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
    def __init__(self, env: Env, agent: Agent, device):
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
                game_advantages.append(rew + self.gamma * new_value - prec_value)
            self.all_advantages.append(game_advantages)

    def compute_gae(self):
        self.all_gae = []
        for game_id in range(len(self.all_advantages)):
            gae = self.all_advantages[game_id][-1]
            cur_gae = []
            for t in range(len(self.all_advantages[game_id]) - 1):
                robot_pos = self.all_unit_pos[game_id][t]
                next_robot_pos = self.all_unit_pos[game_id][t + 1]
                gae = map_robots_grid(robot_pos, next_robot_pos, gae)
                gae = self.all_advantages[game_id][t] + self.gamma * self.lamb * gae
                cur_gae.append(gae)
            self.all_gae.append(list(reversed(cur_gae)))

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
        ret_obs_grid = []
        ret_obs_vector = []
        ret_actions = []
        ret_logprob = []
        ret_gae = []
        ret_masks = []
        ret_returns = []

        for id in ids:
            ret_obs_grid.append(self.all_obs_grid[id[0]][id[1]])
            ret_obs_vector.append(self.all_obs_vectors[id[0]][id[1]])
            ret_actions.append(self.all_actions[id[0]][id[1]])
            ret_logprob.append(self.all_logprob[id[0]][id[1]])
            ret_gae.append(self.all_gae[id[0]][id[1]])
            ret_masks.append(self.all_masks[id[0]][id[1]])
            ret_returns.append(
                self.all_gae[id[0]][id[1]] + self.all_values[id[0]][id[1]]
            )

        ret_obs_grid = th.tensor(ret_obs_grid, device=device, dtype=th.float32)
        ret_obs_vector = th.tensor(ret_obs_vector, device=device, dtype=th.float32)
        ret_actions = th.tensor(ret_actions, device=device, dtype=th.float32)
        ret_logprob = th.tensor(ret_logprob, device=device, dtype=th.float32)
        ret_gae = th.tensor(ret_gae, device=device, dtype=th.float32)
        ret_masks = th.tensor(ret_masks, device=device, dtype=th.float32)
        ret_returns = th.tensor(ret_returns, device=device, dtype=th.float32)

        return (
            ret_obs_grid,
            ret_obs_vector,
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

    def update(self, value):
        self.n += 1
        self.sum += value

    @property
    def value(self):
        return self.sum / self.n


batch_size = 16
mini_batch_size = 4

clip_coef = 0.1
norm_adv = True
ent_coef = 0
vf_coef = 1
max_grad_norm = 0.5

save_path = Path("results/models/survivor")
save_path.mkdir(exist_ok=True, parents=True)


USE_WANDB = False
SAVE_MODEL = False

if __name__ == "__main__":
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    if USE_WANDB:
        wandb.init(project="lux_ai_ppo")

    # env setup
    env = get_env()

    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

    buffer = ReplayBuffer(env, agent, device)

    global_step = 0
    start_time = time.time()

    value_loss = MeanLogger()
    policy_loss = MeanLogger()
    entropy_logger = MeanLogger()
    old_approx_kl_logger = MeanLogger()
    approx_kl_logger = MeanLogger()
    clipfrac_logger = MeanLogger()
    explained_variance = MeanLogger()
    mean_ratio = MeanLogger()

    for update in range(10000):

        start_time = time.time()
        print("Update {} | Trajs computation ".format(update), end="", flush=True)

        if (update) % 10 == 0 and SAVE_MODEL:
            model_name = "{:04d}".format(update)
            th.save(agent.state_dict(), save_path / model_name)

        value_loss.reset()
        policy_loss.reset()
        entropy_logger.reset()
        old_approx_kl_logger.reset()
        approx_kl_logger.reset()
        clipfrac_logger.reset()
        explained_variance.reset()
        mean_ratio.reset()

        buffer.fill(batch_size)
        exit()

        print("| Learning ", end="", flush=True)

        for epoch in range(4):
            for obs_g, obs_v, act, logprob, gae, masks, rets in buffer.sample(
                batch_size=len(buffer) // 4
            ):

                _, newlogprob, entropy, newvalue = agent.get_action(
                    {"grid": obs_g, "vector": obs_v}, masks, act
                )
                logratio = newlogprob - logprob
                ratio = logratio.exp()

                with th.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

                if norm_adv:
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                # Policy loss
                pg_loss1 = -gae * ratio
                pg_loss2 = -gae * th.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = th.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if False:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + th.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - rets) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

                value_loss.update(v_loss.detach().cpu().item())
                policy_loss.update(pg_loss.detach().cpu().item())
                entropy_logger.update(entropy_loss.detach().cpu().item())
                old_approx_kl_logger.update(
                    th.mean(old_approx_kl).detach().cpu().item()
                )
                approx_kl_logger.update(th.mean(approx_kl).detach().cpu().item())
                clipfrac_logger.update(clipfracs)

                y_pred, y_true = (
                    newvalue.detach().cpu().numpy(),
                    rets.detach().cpu().numpy(),
                )
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )
                explained_variance.update(explained_var)
                mean_ratio.update(ratio.mean().detach().cpu().item())

            # exit()

        to_log = {}

        to_log["main/value_loss"] = value_loss.value
        to_log["infos/policy_loss"] = policy_loss.value
        to_log["infos/policy_loss"] = policy_loss.value
        to_log["infos/entropy"] = entropy_logger.value
        to_log["infos/approx_kl"] = approx_kl_logger.value
        to_log["infos/clipfrac"] = clipfrac_logger.value
        to_log["infos/explained_variance"] = explained_variance.value
        to_log["infos/mean_ratio"] = mean_ratio.value

        to_log["main/mean_reward"] = np.mean(list(itertools.chain(*buffer.all_rewards)))

        if USE_WANDB:
            wandb.log(to_log)

        print("| Done ({:.03f}s)".format(time.time() - start_time))

    env.close()
