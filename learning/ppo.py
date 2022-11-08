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


class ObsHead(nn.Module):
    def __init__(self, env: Env):
        super().__init__()

        self.full_grid_size = np.array(env.obs_generator.grid_shape).prod()

        self.grid_head = nn.Sequential(
            nn.Linear(np.array(env.obs_generator.grid_shape).prod(), 32)
        )  # TODO: implement some kind of spatial attention here
        self.vector_head = nn.Sequential(
            nn.Linear(env.obs_generator.vector_shape[0], 32)
        )

    def forward(self, obs):
        grid_obs = obs["grid"].view(-1, self.full_grid_size)
        grid_feat = self.grid_head(grid_obs)

        vector_feat = self.vector_head(obs["vector"])

        return th.relu(th.concat([grid_feat, vector_feat], dim=1))


class Agent(nn.Module):
    def __init__(self, env: Env):
        super().__init__()
        self.act_backbone = ObsHead(env)
        self.critic_backbone = ObsHead(env)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_handler.action_nb), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    # def get_action_and_value(self, x, action=None):
    #     logits = self.actor(x)
    #     probs = Categorical(logits=logits)
    #     if action is None:
    #         action = probs.sample()
    #     return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_action(self, obs, masks, action=None):
        logits = self.actor(self.act_backbone(obs))
        logits = logits - BIG_NUMBER * (1 - masks)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(self.critic_backbone(obs)).view(-1),
        )

    def get_greedy_action(self, obs, masks, action=None):
        logits = self.actor(self.act_backbone(obs))
        logits = logits - BIG_NUMBER * (1 - masks)
        probs = Categorical(logits=logits)
        if action is None:
            action = th.argmax(logits, dim=-1)
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(self.critic_backbone(obs)).view(-1),
        )


def obs_to_network(obs, device):
    grids = []
    vectors = []
    for team in teams:
        for unit_id, unit_obs in obs[team].items():
            grids.append(unit_obs["grid"])
            vectors.append(unit_obs["vector"])

    grid = np.array(grids)
    vectors = np.array(vectors)

    grid = th.tensor(grid, device=device, dtype=th.float32)
    vectors = th.tensor(vectors, device=device, dtype=th.float32)

    return {"grid": grid, "vector": vectors}


def mask_to_network(masks, device):
    to_return = []
    for team in teams:
        for unit_id, unit_mask in masks[team].items():
            to_return.append(unit_mask)

    to_return = np.array(to_return)
    to_return = th.tensor(to_return, device=device, dtype=th.float32)

    return to_return


def actions_to_env(network_actions, obs):
    to_return = {team: {} for team in teams}
    network_actions = network_actions.detach().cpu().numpy()
    id = 0
    for team in teams:
        for unit_id in obs[team].keys():
            to_return[team][unit_id] = network_actions[id]
            id += 1
    return to_return


def multi_agent_rollout(env: Env, agent: Agent, device, max_ep_len=1100):
    obs, masks = env.reset()

    all_obs = [obs]
    all_rewards = []
    all_masks = [masks]
    all_actions = []
    all_log_prob = []
    all_values = []

    no_unit = False

    for t in range(max_ep_len):
        network_obs = obs_to_network(obs, device)
        # no_unit = no_unit or network_obs["grid"].shape[0] == 0

        if network_obs["grid"].shape[0] > 0:
            network_masks = mask_to_network(masks, device)
            actions, log_prob, _, value = agent.get_action(network_obs, network_masks)
            actions = actions_to_env(actions, obs)
            log_prob = actions_to_env(log_prob, obs)
            value = actions_to_env(value, obs)
        else:
            actions = {team: {} for team in teams}
            log_prob = {team: {} for team in teams}
            value = {team: {} for team in teams}

        obs, rewards, masks, done = env.step(actions)

        all_values.append(value)

        if done:
            break

        all_obs.append(obs)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_masks.append(masks)
        all_log_prob.append(log_prob)

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
    }


class ReplayBuffer:
    def __init__(self, env: Env, agent: Agent, device):
        self.env = env
        self.agent = agent
        self.device = device

        self.gamma = 0.99
        self.lamb = 0.95

    def fill(self, batch_size: int):
        self.all_obs_grid = []
        self.all_obs_vectors = []
        self.all_actions = []
        self.all_rewards = []
        self.all_masks = []
        self.all_logprob = []
        self.all_values = []

        while len(self) < batch_size:
            self.expand(multi_agent_rollout(self.env, self.agent, self.device))

    def expand(self, rollout):
        for team in teams:
            all_agents = set()
            for obs in rollout["obs"]:
                for unit_id in obs[team].keys():
                    all_agents.add(unit_id)
            all_agents = sorted(all_agents, key=lambda x: int(x[5:]))

            for agent_id in all_agents:
                agent_obs_grid = []
                agent_obs_vector = []
                agent_actions = []
                agent_rewards = []
                agent_actmasks = []
                agent_logprob = []
                agent_values = []

                for obs in rollout["obs"]:
                    if agent_id in obs[team]:
                        agent_obs_grid.append(obs[team][agent_id]["grid"])
                        agent_obs_vector.append(obs[team][agent_id]["vector"])
                for act in rollout["actions"]:
                    if agent_id in act[team]:
                        agent_actions.append(act[team][agent_id])
                for rew in rollout["rewards"]:
                    if agent_id in rew[team]:
                        agent_rewards.append(rew[team][agent_id])
                for masks in rollout["masks"]:
                    if agent_id in masks[team]:
                        agent_actmasks.append(masks[team][agent_id])
                for logprob in rollout["logprob"]:
                    if agent_id in logprob[team]:
                        agent_logprob.append(logprob[team][agent_id])
                for values in rollout["values"]:
                    if agent_id in values[team]:
                        agent_values.append(values[team][agent_id])

                self.all_obs_grid.append(agent_obs_grid)
                self.all_obs_vectors.append(agent_obs_vector)
                self.all_actions.append(agent_actions)
                self.all_rewards.append(agent_rewards)
                self.all_masks.append(agent_actmasks)
                self.all_logprob.append(agent_logprob)
                self.all_values.append(agent_values)

        self.compute_advantage()
        self.compute_gae()

    def compute_advantage(self):
        self.all_advantages = []
        for i in range(len(self.all_rewards)):
            cur_advantages = []
            for t in range(len(self.all_rewards[i])):
                rew = self.all_rewards[i][t]
                prec_value = self.all_values[i][t]
                if t + 1 < len(self.all_values[i]):
                    new_value = self.all_values[i][t + 1]
                else:
                    new_value = 0
                cur_advantages.append(rew + self.gamma * new_value - prec_value)
            self.all_advantages.append(cur_advantages)

    def compute_gae(self):
        self.all_gae = []
        for i in range(len(self.all_advantages)):
            cur_gae = []
            gae = 0
            for t in reversed(range(len(self.all_advantages[i]))):
                gae = self.all_advantages[i][t] + self.gamma * self.lamb * gae
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


USE_WANDB = True
SAVE_MODEL = True

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
