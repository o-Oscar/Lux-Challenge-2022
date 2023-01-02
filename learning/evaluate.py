import argparse
from pathlib import Path

import torch as th

from utils.bots import Bot
from utils.env import Env

from learning.ppo import multi_agent_rollout


def evaluate(args):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    bot = Bot(args.bot_type, args.vec_chan)
    agent = bot.agent

    env = Env(bot.action, bot.obs_generator, bot.reward_generators[-1])

    if len(bot.reward_update_nbs) > 1:
        if args.num_reward == -1:
            num_reward = len(bot.reward_update_nbs)
        save_path = (
            Path("results") / args.bot_type / ((args.name) + "_rew_" + str(num_reward))
        )
        replay_name = args.bot_type + "_" + args.name + "_rew_" + str(num_reward)

    else:
        save_path = Path("results") / args.bot_type / args.name
        replay_name = args.bot_type + "_" + args.name

    save_path_model = list(sorted(save_path.glob("*")))[-1]

    agent.load_state_dict(th.load(save_path_model))
    agent.to(device)
    agent.eval()

    multi_agent_rollout(env, agent, device, replay_name=replay_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bot_type",
        type=str,
        default="default",
        help="Type of the Bot.",
    )

    parser.add_argument(
        "--vec_chan",
        type=int,
        default=32,
        help="Number of output channels for the vector Obs Head",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Name of the run. Used to get the last version of the bot.",
    )

    parser.add_argument(
        "--num_reward",
        type=int,
        default=-1,
        help="Number of the reward to evaluate for this bot",
    )

    args = parser.parse_args()

    evaluate(args)
