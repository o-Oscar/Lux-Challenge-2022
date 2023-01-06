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
from utils import teams


def evaluate(args):
    device = th.device("cpu")

    agent = BaselineAgent()

    env = Env(
        HarvestTransferActionHandler(),
        PositionIceFactoryObsGenerator(),
        FactorySurvivorRewardGenerator(),
        water_consumption = args.water_consumption
    )

    replay_name = "baseline"
    mean_reward = 0
    for _ in tqdm.trange(args.num_games):
        res_game = multi_agent_rollout(
            env, agent, device, replay_name=replay_name, max_ep_len=args.max_length
        )
        for turn in range(len(res_game["rewards_monitoring"])):
            nb_factories = res_game["nb_factories"][turn]
            for team in teams:
                m = res_game["masks"][turn][team]
                m = np.max(m, axis=0)
                r = res_game["rewards_monitoring"][turn][team]

                mean_reward += (
                    np.sum(m * r) / nb_factories / len(teams) / args.num_games
                )
    print("Mean transfer per factory and per game :", mean_reward)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

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
        help="Number of the reward to evaluate for this bot",
    )

    args = parser.parse_args()

    evaluate(args)
