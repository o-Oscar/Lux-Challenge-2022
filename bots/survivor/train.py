import argparse
from pathlib import Path

import numpy as np
import torch as th

from bots.survivor import ACTION_HANDLER, OBS_GENERATOR, REWARD_GENERATOR
from bots.survivor.agent import Agent
from learning.ppo import PPOConfig, start_ppo
from utils.env import Env


def train(args):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    env = Env(ACTION_HANDLER, OBS_GENERATOR, REWARD_GENERATOR)
    agent = Agent(env)

    config = PPOConfig(
        env=env,
        agent=agent,
        save_path=Path("results/survivor") / args.name,
        wandb=args.wandb,
        epoch_per_save=args.epoch_per_save,
        device=device,
        min_batch_size=args.min_batch_size,
    )

    start_ppo(config)


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
