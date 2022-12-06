import argparse
from pathlib import Path

import numpy as np
import torch as th

from bots.thirsty_dqn.agent import Agent
from learning.dqn import DqnConfig, start_dqn
from utils.action.move import MoveActionHandler
from utils.env import Env
from utils.obs.complete import CompleteObsGenerator
from utils.reward.survivor import SurvivorRewardGenerator
from utils.reward.thirsty import ThirstyReward


def train(args):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    action_handler = MoveActionHandler()
    obs_generator = CompleteObsGenerator()
    reward_generator = ThirstyReward()
    # reward_generator = SurvivorRewardGenerator()

    env = Env(action_handler, obs_generator, reward_generator)

    agent = Agent(env, device)
    target_agent = Agent(env, device)

    config = DqnConfig(
        env=env,
        agent=agent,
        target_agent=target_agent,
        save_path=Path("results/models") / args.name,
        wandb=args.wandb,
        epoch_per_save=args.epoch_per_save,
        device=device,
    )

    start_dqn(config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to use wandb or not.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="default",
        help="Name of the run. Used for saving and wandb name.",
    )
    parser.add_argument(
        "--epoch_per_save",
        type=int,
        default=0,
        help="Number of epoch between saves.",
    )

    parser.add_argument(
        "--min_batch_size",
        type=int,
        default=32,
        help="Min Batch Size.",
    )

    args = parser.parse_args()

    train(args)
