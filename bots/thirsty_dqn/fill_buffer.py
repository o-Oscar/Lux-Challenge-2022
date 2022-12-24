import argparse
import pickle
from pathlib import Path

import numpy as np
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
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    action_handler = MoveActionHandler()
    obs_generator = CompleteObsGenerator()
    reward_generator = GlobalSurvivorRewardGenerator()

    env = Env(action_handler, obs_generator, reward_generator)

    # create a buffer object
    buffer = ReplayBuffer()

    # create the agent
    agent = Agent(env, device).to(device)
    network_save_path: Path = Path("results/thirsty_dqn/agents") / args.network
    agent.load_state_dict(th.load(network_save_path))

    while len(buffer) < args.steps:
        print("Starting new game")

        rollout = multi_agent_rollout(
            env, agent, device, args.max_rollout_len, 0.0001, 0.1
        )

        max_robot_nb = {team: 0 for team in teams}
        for unit_pos in rollout["unit_pos"]:
            for team in teams:
                max_robot_nb[team] = max(max_robot_nb[team], len(unit_pos[team]))
        final_robot_nb = {team: len(rollout["unit_pos"][-1][team]) for team in teams}

        print("max_robot_nb :", max_robot_nb)
        print("final_robot_nb :", final_robot_nb)

        # print the max number of robot and the final number of robots

        buffer.expand(rollout)

    # save the buffer
    buffer_save_path: Path = Path("results/thirsty_dqn/buffers") / args.out
    buffer_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(buffer_save_path, "wb") as f:
        pickle.dump(buffer, f)
    print("Buffer saved at {}".format(buffer_save_path))


"""
Usage :
python -m bots.thirsty_dqn.fill_buffer --network 0000 --out 0000 --steps 6000 --max_rollout_len 30
python -m bots.thirsty_dqn.fill_buffer --network 0001 --out 0001 --steps 6000 --max_rollout_len 30
python -m bots.thirsty_dqn.fill_buffer --network 0002 --out 0002 --steps 6000 --max_rollout_len 30
python -m bots.thirsty_dqn.fill_buffer --network notargetupdate2 --out test --steps 1000 --max_rollout_len 30
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        help="Network file to use for as rollout agent.",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Out save file for the buffer.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Number of steps to add to the buffer.",
    )
    parser.add_argument(
        "--max_rollout_len",
        type=int,
        default=1100,
        help="Max number of env steps per game.",
    )

    args = parser.parse_args()

    fill_buffer(args)
