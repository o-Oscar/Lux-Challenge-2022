import argparse

import numpy as np
import torch as th
import tqdm

from learning.ppo import multi_agent_rollout

from utils.env import Env
from utils.action.harvest_transfer import HarvestTransferActionHandler
from utils.obs.position_ice_factory import PositionIceFactoryObsGenerator
from utils.reward.factory_survivor import FactorySurvivorRewardGenerator
from utils.agent.baseline import BaselineAgent
from utils.agent.null import NullAgent

from utils import teams

import scipy.stats
import time


def evaluate(args):
    device = th.device("cpu")

    if args.null_agent:
        agent = NullAgent()
    else:
        agent = BaselineAgent()

    env = Env(
        HarvestTransferActionHandler(),
        PositionIceFactoryObsGenerator(),
        FactorySurvivorRewardGenerator(),
        water_consumption=args.water_consumption,
    )

    replay_name = "baseline"

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
        "--water_consumption",
        type=int,
        default=0,
        help="Water consumption of factories",
    )

    parser.add_argument(
        "--null_agent",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wether to use a the null agent or not",
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
