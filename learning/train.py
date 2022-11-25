import argparse
from pathlib import Path

import torch as th

from utils.bots import Bot
from utils.env import Env

from learning.ppo import PPOConfig, start_ppo


def train(args):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    bot = Bot(args.bot_type)
    agent = bot.agent

    if args.init_save_path is not None:
        agent.load_state_dict(th.load(args.init_save_path))

    if len(bot.reward_update_nbs) > 1:  # case of multi training
        for i, (reward_generator, reward_update_nb) in enumerate(
            zip(
                bot.reward_generators[args.skip_rewards :],
                bot.reward_update_nbs[args.skip_rewards :],
            )
        ):
            env = Env(bot.action, bot.obs_generator, reward_generator)

            config = PPOConfig(
                env=env,
                agent=agent,
                save_path=Path("results")
                / args.bot_type
                / ((args.name) + "_rew_" + str(i + 1 + args.skip_rewards)),
                name=args.bot_type + "_" + args.name,
                wandb=args.wandb,
                epoch_per_save=args.epoch_per_save,
                device=device,
                min_batch_size=args.min_batch_size,
                update_nb=reward_update_nb,
            )
            start_ppo(config)
    else:
        env = Env(bot.action, bot.obs_generator, bot.reward_generators[0])

        config = PPOConfig(
            env=env,
            agent=agent,
            save_path=Path("results") / args.bot_type / (args.name),
            name=args.bot_type + "_" + args.name,
            wandb=args.wandb,
            epoch_per_save=args.epoch_per_save,
            device=device,
            min_batch_size=args.min_batch_size,
            update_nb=bot.reward_update_nbs[0],
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
        "--bot_type",
        type=str,
        default="default",
        help="Type of the Bot.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Name of the run. Used for saving and wandb name in addition to the bot_type.",
    )

    parser.add_argument(
        "--epoch_per_save",
        type=int,
        default=0,
        help="Number of epoch between saves.",
    )

    parser.add_argument(
        "--init_save_path",
        type=str,
        default=None,
        help="Path to the pretrained agent. It is use if it is not None",
    )

    parser.add_argument(
        "--skip_rewards",
        type=int,
        default=0,
        help="Number of rewards to skip in the training of the bot",
    )

    parser.add_argument(
        "--min_batch_size",
        type=int,
        default=1,
        help="Mini Batch Size",
    )

    args = parser.parse_args()

    train(args)
