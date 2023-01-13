import argparse
from pathlib import Path

import numpy as np
import torch as th

from utils.bots import Bot
from utils.env import Env
from utils import teams
import tqdm

from learning.ppo import multi_agent_rollout

import scipy.stats
import time


def evaluate(args):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    bot = Bot(args.bot_type, args.vec_chan)
    agent = bot.agent

    env = Env(
        bot.action,
        bot.obs_generator,
        bot.reward_generators[-1],
        water_consumption=args.water_consumption,
    )

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

    mean_rewards = []
    times = []
    for _ in tqdm.trange(args.num_games):
        begin = time.time()
        res_game = multi_agent_rollout(
            env, agent, device, replay_name=replay_name, max_ep_len=args.max_length
        )
        end = time.time()
        times.append(end - begin)
        mean_reward = 0
        for turn in range(len(res_game["rewards_monitoring"])):
            nb_factories = res_game["nb_factories"][turn]
            for team in teams:
                m = res_game["masks"][turn][team]
                m = np.max(m, axis=0)
                r = res_game["rewards_monitoring"][turn][team]
                mean_reward += np.sum(m * r) / nb_factories / len(teams)
        mean_rewards.append(mean_reward)

    mean_rewards = np.array(mean_rewards)
    times = np.array(times)
    m_reward, h_reward = mean_confidence_interval(mean_rewards)
    m_time, h_time = mean_confidence_interval(times)
    print("Mean transfer per factory and per game :", m_reward, "+/-", h_reward)
    print("Mean time per per game :", m_time, "+/-", h_time)


def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    m, se = np.mean(data), scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


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

    parser.add_argument(
        "--water_consumption",
        type=int,
        default=0,
        help="Water consumption of factories",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=1000,
        help="Max length of a game",
    )

    parser.add_argument(
        "--num_games",
        type=int,
        default=1,
        help="Number games done to evaluate this bot",
    )

    args = parser.parse_args()

    evaluate(args)
