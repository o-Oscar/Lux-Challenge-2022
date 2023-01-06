import argparse
from pathlib import Path

import torch as th

from utils.bots import Bot
from utils.env import Env

from learning.ppo import PPOConfig, start_ppo


def train(args):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    bot = Bot(args.bot_type, args.vec_chan, args.use_relu)
    agent = bot.agent

    if args.init_save_path is not None:
        agent.load_state_dict(th.load(args.init_save_path))

    # Case of Sequential Training (multiple rewards)
    if len(bot.reward_update_nbs) > 1:
        for i, (reward_generator, reward_update_nb) in enumerate(
            zip(
                bot.reward_generators[args.skip_rewards :],
                bot.reward_update_nbs[args.skip_rewards :],
            )
        ):
            env = Env(
                bot.action,
                bot.obs_generator,
                reward_generator,
                water_consumption=args.water_consumption,
                max_length=args.max_length,
            )

            config = PPOConfig(
                env=env,
                agent=agent,
                save_path=Path("results")
                / args.bot_type
                / ((args.name) + "_rew_" + str(i + 1 + args.skip_rewards)),
                name=args.bot_type + "_" + args.name,
                wandb=args.wandb,
                run_id=args.run_id,
                epoch_per_save=args.epoch_per_save,
                device=device,
                min_batch_size=args.min_batch_size,
                learning_batch_size=args.learning_batch_size,
                update_nb=reward_update_nb,
                gamma=args.gamma,
            )
            start_ppo(config)

    else:
        env = Env(
            bot.action,
            bot.obs_generator,
            bot.reward_generators[0],
            water_consumption=args.water_consumption,
            max_length=args.max_length,
        )

        config = PPOConfig(
            env=env,
            agent=agent,
            save_path=Path("results") / args.bot_type / (args.name),
            name=args.bot_type + "_" + args.name,
            wandb=args.wandb,
            run_id=args.run_id,
            epoch_per_save=args.epoch_per_save,
            device=device,
            min_batch_size=args.min_batch_size,
            learning_batch_size=args.learning_batch_size,
            update_nb=bot.reward_update_nbs[0],
            gamma=args.gamma,
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
        "--run_id",
        type=str,
        default=None,
        help="Wandb run id to retake training",
    )

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
        "--use_relu",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wether to use a ReLU or not",
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

    parser.add_argument(
        "--learning_batch_size",
        type=int,
        default=300,
        help="Learning Batch Size",
    )

    parser.add_argument(
        "--water_consumption",
        type=int,
        default=1,
        help="Water consumption of factories",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=1000,
        help="Max length of a game",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )

    args = parser.parse_args()

    train(args)
