import time

import gym
import numpy as np
import torch as th
import torch.nn as nn
from utils.env import get_env, Env
from utils import teams
import itertools
import wandb
from pathlib import Path
from learning.ppo import (
    Agent,
    obs_to_network,
    mask_to_network,
    actions_to_env,
    multi_agent_rollout,
)


def multi_agent_rollout_deterministic(env: Env, agent: Agent, max_ep_len=1100):
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
            actions, log_prob, _, value = agent.get_greedy_action(
                network_obs, network_masks
            )
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


save_path = Path("results/models/survivor")


MC_NB = 4

if __name__ == "__main__":

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    env = get_env()

    agent = Agent(env).to(device)

    all_model_names = sorted(save_path.glob("*"))

    all_test_models = all_model_names[::5]

    for model_name in all_test_models:
        print("Test model name :", model_name)

        agent.load_state_dict(th.load(model_name))
        agent.eval()

        for i in range(MC_NB):
            rollout = multi_agent_rollout_deterministic(env, agent)
            # rollout = multi_agent_rollout(env, agent, device)

            all_agents = set()
            for team in teams:
                for obs in rollout["obs"]:
                    for unit_id in obs[team].keys():
                        all_agents.add(unit_id)

            all_agents = sorted(all_agents, key=lambda x: int(x[5:]))

            surviving_agents = set()
            for team in teams:
                for unit_id in rollout["obs"][-1][team].keys():
                    surviving_agents.add(unit_id)
            surviving_agents = sorted(surviving_agents, key=lambda x: int(x[5:]))

            print(
                "Survivors : {}/{} = {:.02f}".format(
                    len(surviving_agents),
                    len(all_agents),
                    len(surviving_agents) / len(all_agents),
                )
            )
