import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th

from bots.thirsty_dqn.agent import Agent
from learning.dqn import multi_agent_rollout
from learning.dqn_buffer import ReplayBuffer
from utils import teams
from utils.action.move import MoveActionHandler
from utils.env import Env
from utils.obs.complete import CompleteObsGenerator
from utils.reward.survivor import GlobalSurvivorRewardGenerator


def fill_buffer(args):
    tau = 0.0001
    epsilon = 0.0

    network_dir = Path("results/thirsty_dqn/agents")
    all_net_paths = sorted(list(network_dir.glob(args.networks + "_*")))
    # print(all_nets[0].name)
    # exit()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    action_handler = MoveActionHandler()
    obs_generator = CompleteObsGenerator()
    reward_generator = GlobalSurvivorRewardGenerator()

    env = Env(action_handler, obs_generator, reward_generator)

    # create the agent

    all_df = []
    mean_df = []
    for path in all_net_paths:
        print("testing agent {}".format(path.name))
        agent = Agent(env, device).to(device)
        agent.load_state_dict(th.load(path))
        to_save = test_agent(agent, env, device, tau, epsilon)
        df = pd.DataFrame(to_save)
        mean = {
            key: df[key].mean()
            for key in [
                "max_robot_nb_0",
                "max_robot_nb_1",
                "final_robot_nb_0",
                "final_robot_nb_1",
            ]
        }
        mean["network"] = path.name
        df["network"] = path.name
        all_df.append(df)
        mean_df.append(mean)
        print(mean)

    mean_df = pd.DataFrame(mean_df)
    print(mean_df)
    mean_df.to_csv("results/thirsty_dqn/network_perf.csv", index=False)


def test_agent(agent, env, device, tau, epsilon):
    all_infos = []
    for game_id in range(args.games):
        print("Starting new game")

        rollout = multi_agent_rollout(
            env, agent, device, tau, epsilon, args.max_rollout_len
        )

        max_robot_nb = {team: 0 for team in teams}
        for unit_pos in rollout["unit_pos"]:
            for team in teams:
                max_robot_nb[team] = max(max_robot_nb[team], len(unit_pos[team]))
        final_robot_nb = {team: len(rollout["unit_pos"][-1][team]) for team in teams}

        to_save = {
            "max_robot_nb_0": max_robot_nb["player_0"],
            "max_robot_nb_1": max_robot_nb["player_1"],
            "final_robot_nb_0": final_robot_nb["player_0"],
            "final_robot_nb_1": final_robot_nb["player_1"],
        }
        print(to_save)
        all_infos.append(to_save)

    return all_infos


"""
Usage :
python -m bots.thirsty_dqn.test_networks --networks a --games 5 --max_rollout_len 30
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--networks",
        type=str,
        help="Network file to use for as rollout agent.",
    )
    parser.add_argument(
        "--games",
        type=int,
        help="Number of games to used for testing.",
    )
    parser.add_argument(
        "--max_rollout_len",
        type=int,
        default=1100,
        help="Max number of env steps per game.",
    )

    args = parser.parse_args()

    fill_buffer(args)
