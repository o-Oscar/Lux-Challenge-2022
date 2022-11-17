import argparse
from pathlib import Path

import numpy as np
import torch as th

from bots.survivor_position import ACTION_HANDLER, OBS_GENERATOR, REWARD_GENERATOR
from bots.survivor_position.agent import Agent
from learning.ppo import multi_agent_rollout
from utils.env import Env


def evaluate(args):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    env = Env(ACTION_HANDLER, OBS_GENERATOR, REWARD_GENERATOR)
    agent = Agent(env)
    save_path = Path("results/") / "survivor_position" / args.name

    save_path_model = list(sorted(save_path.glob("*"), reverse=True))[0]

    agent.load_state_dict(th.load(save_path_model))
    agent.to(device)
    agent.eval()

    multi_agent_rollout(env, agent, device, replay_name=args.name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="default",
        help="Name of the run. Used for saving and wandb name.",
    )

    args = parser.parse_args()

    evaluate(args)
