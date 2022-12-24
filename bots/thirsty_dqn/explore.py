import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from bots.thirsty_dqn.agent import Agent
from learning.dqn import ReplayBuffer
from utils import teams
from utils.action.move import MoveActionHandler
from utils.env import Env
from utils.obs.complete import CompleteObsGenerator
from utils.reward.thirsty import ThirstyReward

# init an env with known setup

if False:
    with open("coucou.pkl", "rb") as f:
        buffer: ReplayBuffer = pickle.load(f)

    for i in range(len(buffer.all_masks)):
        for t in range(len(buffer.all_masks[i])):
            if buffer.all_masks[i][t][0, 10, 12]:
                print(
                    buffer.all_actions[i][t][10, 12], buffer.all_rewards[i][t][10, 12]
                )

    exit()


action_handler = MoveActionHandler()
obs_generator = CompleteObsGenerator()
reward_generator = ThirstyReward()

env = Env(action_handler, obs_generator, reward_generator)

device = th.device("cuda" if th.cuda.is_available() else "cpu")
agent = Agent(env, device).to(device)
# agent.q_network.load_state_dict(th.load("results/models/default_init/0000"))
agent.q_network.load_state_dict(th.load("results/models/default/q_network_0000"))
agent.v_network.load_state_dict(th.load("results/models/default/v_network_0000"))

full_obs, masks, robot_pos = env.reset(seed=42)


def get_q_values(obs, mask, x, y, action=None):
    obs = th.tensor(obs, dtype=th.float32, device=device).view(1, 11, 48, 48)
    mask_np = mask
    mask_th = th.tensor(mask_np, dtype=th.float32, device=device).view(1, 5, 48, 48)

    if action is None:
        cur_action_th = th.zeros((5, 48, 48), dtype=th.long, device=device)
    else:
        cur_action_th = (
            th.tensor(action, dtype=th.long, device=device)
            .view(1, 48, 48)
            .repeat((5, 1, 1))
        )
    for i in range(5):
        cur_action_th[i, x, y] = i

    to_return = agent.q_eval(
        obs.repeat((5, 1, 1, 1)), cur_action_th, mask_th.repeat((5, 1, 1, 1))
    )
    return to_return[:, x, y].detach().cpu().numpy()


def get_probs(obs, mask, x, y, tau):
    q_values = get_q_values(obs, mask, x, y)
    tau = 0.03
    eq = np.exp((q_values - np.max(q_values)) / tau)
    probs = eq / np.sum(eq)
    return probs


with open("coucou.pkl", "rb") as f:
    buffer: ReplayBuffer = pickle.load(f)

# for i in range(len(buffer.all_masks)):
for i in [66]:
    for t in range(len(buffer.all_masks[i])):
        # for t in [10]:
        if buffer.all_masks[i][t][0, 10, 12]:
            print(buffer.all_actions[i][t][10, 12], buffer.all_rewards[i][t][10, 12])
            print(
                get_q_values(
                    buffer.all_obs[i][t],
                    buffer.all_masks[i][t],
                    10,
                    12,
                    # buffer.all_actions[i][t],
                )
            )

            obs = th.tensor(buffer.all_obs[i][t], dtype=th.float32, device=device).view(
                1, 11, 48, 48
            )
            mask_th = th.tensor(
                buffer.all_masks[i][t], dtype=th.float32, device=device
            ).view(1, 5, 48, 48)
            action_th = th.tensor(
                buffer.all_actions[i][t], dtype=th.long, device=device
            ).view(1, 48, 48)
            print(agent.q_eval(obs, action_th, mask_th)[0, 10, 12].item())
            print(i, t)
            print()
            plt.imshow(buffer.all_masks[i][t][0])
            plt.show()
        # plt.imshow(buffer.all_obs[i][t][4])
        # plt.show()
        # exit()
exit()

for i in range(10):

    # print(get_probs(full_obs["player_0"], masks["player_0"], 10, 12, 0.03))
    # print(get_probs(full_obs["player_0"], masks["player_0"], 10, 11, 0.03))
    print(get_q_values(full_obs["player_0"], masks["player_0"], 10, 12))
    print(get_q_values(full_obs["player_0"], masks["player_0"], 10, 11))
    print()

    a = np.zeros((48, 48), dtype=np.int32)
    if i == 0:
        a[10, 12] = 1
    elif i == 1:
        a[10, 11] = 4
    action = {team: a for team in teams}
    full_obs, rewards, masks, done, units_pos = env.step(action)
    if done:
        break

env.save(full_save=False, convenient_save=True, convenient_name="explore")
