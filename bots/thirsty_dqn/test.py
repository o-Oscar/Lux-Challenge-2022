import numpy as np
import torch as th

from bots.thirsty_dqn.agent import Agent
from utils.action.move import MoveActionHandler
from utils.env import Env
from utils.obs.complete import CompleteObsGenerator
from utils.reward.thirsty import ThirstyReward

# init an env with known setup

action_handler = MoveActionHandler()
obs_generator = CompleteObsGenerator()
reward_generator = ThirstyReward()

env = Env(action_handler, obs_generator, reward_generator)

full_obs, masks, robot_pos = env.reset(seed=42)
grid_action = np.zeros((48, 48))

# compute the q value
device = th.device("cuda" if th.cuda.is_available() else "cpu")
agent = Agent(env, device).to(device)
agent.load_state_dict(th.load("results/models/default/0000"))

obs = th.tensor(full_obs["player_0"], dtype=th.float32, device=device).view(
    1, 11, 48, 48
)
masks = th.tensor(masks["player_0"], dtype=th.float32, device=device).view(1, 5, 48, 48)
act = th.zeros((1, 48, 48), dtype=th.float32, device=device)
act[0, 16, 11] = 0

print(obs.shape, act.shape)

print(agent.q_eval(obs, act, masks)[0, 16, 11])
