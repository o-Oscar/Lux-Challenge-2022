import matplotlib.pyplot as plt
import numpy as np
import torch as th

from bots.thirsty_dqn.agent import Agent
from utils import teams
from utils.action.move import MoveActionHandler
from utils.env import Env
from utils.obs.complete import CompleteObsGenerator
from utils.reward.thirsty import ThirstyReward

# init an env with known setup

action_handler = MoveActionHandler()
obs_generator = CompleteObsGenerator()
reward_generator = ThirstyReward()

env = Env(action_handler, obs_generator, reward_generator)

device = th.device("cuda" if th.cuda.is_available() else "cpu")
agent = Agent(env, device).to(device)
# agent.q_network.load_state_dict(th.load("results/models/default_init/0000"))
agent.q_network.load_state_dict(th.load("results/models/default/q_network_0000"))

# compute the q value for some action (example)
if True:
    full_obs, masks, robot_pos = env.reset(seed=42)

    obs = th.tensor(full_obs["player_0"], dtype=th.float32, device=device).view(
        1, 11, 48, 48
    )
    masks = th.tensor(masks["player_0"], dtype=th.float32, device=device).view(
        1, 5, 48, 48
    )
    act = th.zeros((1, 48, 48), dtype=th.float32, device=device)
    act[0, 16, 11] = 0
    print(agent.q_eval(obs, act, masks)[0, 16, 11])

    act = th.zeros((1, 48, 48), dtype=th.float32, device=device)
    act[0, 16, 11] = 1
    print(agent.q_eval(obs, act, masks)[0, 16, 11])
    exit()


def refine_actions(agent, obs, init_action, mask, mask_np):
    possible_actions = np.where(mask_np)
    if np.sum(mask_np) == 0:
        return init_action
    all_test_actions = []
    for a, x, y in zip(*possible_actions):
        cur_action = th.clone(init_action)
        cur_action[0, x, y] = a
        all_test_actions.append(cur_action)

    all_test_actions = th.concat(all_test_actions, dim=0)
    rep_obs = obs.repeat((all_test_actions.shape[0], 1, 1, 1))
    rep_mask = mask.repeat((all_test_actions.shape[0], 1, 1, 1))
    q_values = agent.q_eval(rep_obs, all_test_actions, rep_mask)

    to_return = th.clone(init_action)
    best_q = np.zeros((48, 48)) - 100000
    for i, (a, x, y) in enumerate(zip(*possible_actions)):
        cur_q = q_values[i, x, y]
        if cur_q > best_q[x, y]:
            best_q[x, y] = cur_q
            to_return[0, x, y] = a
        # if x == 16 and y == 11:
        #     print(a, cur_q)

    return to_return


def choose_action(agent, full_obs, masks):
    to_return = {}
    for team in teams:
        obs = th.tensor(full_obs[team], dtype=th.float32, device=device).view(
            1, 11, 48, 48
        )
        cur_action_th = th.zeros((1, 48, 48), dtype=th.long, device=device)
        mask_np = masks[team]
        mask_th = th.tensor(mask_np, dtype=th.float32, device=device).view(1, 5, 48, 48)

        # we sould repeat this line a few times. But let's not go there yet
        for i in range(10):
            cur_action_th = refine_actions(agent, obs, cur_action_th, mask_th, mask_np)
            # if team == "player_0":
            #     print(cur_action_th[0, 16, 11])
        to_return[team] = cur_action_th[0].detach().cpu().numpy()

    return to_return


def choose_action_random(agent, full_obs, masks):
    to_return = {}
    for team in teams:
        obs = th.tensor(full_obs[team], dtype=th.float32, device=device).view(
            1, 11, 48, 48
        )
        mask_np = masks[team]
        mask_th = th.tensor(mask_np, dtype=th.float32, device=device).view(1, 5, 48, 48)

        cur_action_th = agent.sample_actions(obs, mask_th, 0.01)  # 0.01 for max
        to_return[team] = cur_action_th[0].detach().cpu().numpy()

    return to_return


full_obs, masks, robot_pos = env.reset(seed=42)
for i in range(1100):
    print(i)
    # action = choose_action_random(agent, full_obs, masks)
    action = choose_action(agent, full_obs, masks)
    full_obs, rewards, masks, done, units_pos = env.step(action)
    if done:
        break
print("success !!")

env.save(full_save=False, convenient_save=True, convenient_name="test")


# for each robot
# for each possible action
# create the action array on which to run the network

# run the network on each action

# for each team
# for each robot
# look at the best action designated by the model
# select this action

# start from the top
