import argparse
import dataclasses
from enum import Enum
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions.categorical import Categorical


class States(Enum):
    START = 1
    HOLD = 2
    END = 3


class PrisonerEnv:
    def __init__(self, K=3):
        self.K = K
        self.r = 1
        self.gamma = 1

    def reset(self):
        self.state = States.START
        return np.array([1, 0, 0, 0])

    def step(self, actions):
        actions = actions[:, 0]
        if self.state == States.START:
            self.state = States.HOLD
            if np.all(actions == 1):
                self.to_reward = self.r
            else:
                self.to_reward = -np.sum(actions)
            return np.array([0, 1, 0, self.to_reward]), 0, False
        elif self.state == States.HOLD:
            self.state = States.END
            return np.array([0, 0, 1, 0]), self.to_reward, True
        else:
            raise ValueError("Calling .step() on a terminated env")


def cat_sample(inps):
    probs = Categorical(probs=inps)
    return nn.functional.one_hot(probs.sample(), num_classes=2).type(th.float32)


def apply_qt(inp, t):
    e = th.exp(-t)
    f = (1 - e) / 2
    s = th.sum(inp, dim=-1, keepdim=True)
    return e * inp + f * s


class Agent(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        # self.actor = nn.Sequential(nn.Linear(11, 64), nn.ReLU(), nn.Linear(64, 6))
        # self.critic = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
        self.actor = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )
        self.critic = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def calc_logits(self, s: th.Tensor, a: th.Tensor, t: th.Tensor):
        s_inp = s
        a_inp = a.view((a.shape[0], 6))
        t_inp = th.ones((a.shape[0], 1), device=self.device) * th.log(t)

        inp = th.concat([s_inp, a_inp, t_inp], dim=1)
        logits = self.actor(inp).view((a.shape))
        return logits

    def get_np_action(self, s: np.array):
        s_inp = th.tensor(s, dtype=th.float32, device=self.device).view((1, 4))
        a = self.sample(s_inp)
        return a.detach().cpu().numpy()[0]

    def sample(self, s, num=10):
        all_t = np.logspace(-2, 0, num=num)

        cur_a = cat_sample(th.ones(s.shape[0], 3, 2).to(self.device) / 2)

        all_x0 = []
        for i in range(2):
            x0 = np.zeros((1, 3, 2))
            x0[0, :, i] = 1

            all_x0.append(th.tensor(x0, dtype=th.float32, device=self.device))

        for i in range(len(all_t) - 1, -1, -1):
            if i == 0:
                dt = all_t[i]
            else:
                dt = all_t[i] - all_t[i - 1]

            t = th.tensor(all_t[i], dtype=th.float32, device=self.device)
            dt = th.tensor(dt, dtype=th.float32, device=self.device)

            logits = self.calc_logits(s, cur_a, t)
            probs = th.exp(logits) / th.sum(th.exp(logits), dim=-1, keepdim=True)

            new_x_probs = 0
            for x0 in all_x0:
                p_x0 = th.sum(probs * x0, dim=-1, keepdims=True)
                sum = th.sum(x0 * apply_qt(cur_a, t), dim=-1, keepdim=True)
                new_x_probs = (
                    new_x_probs
                    + p_x0 * (apply_qt(cur_a, dt) * apply_qt(x0, t - dt)) / sum
                )
            cur_a = cat_sample(new_x_probs)

        return cur_a

    def calc_q(self, s, a):
        s_inp = s
        a_inp = a.view((a.shape[0], 6))

        inp = th.concat([s_inp, a_inp], dim=1)
        return self.critic(inp).view((a.shape[0]))

    def get_np_action(self, s: np.array):
        s_inp = th.tensor(s, dtype=th.float32, device=self.device).view((1, 4))
        a = self.sample(s_inp)
        return a.detach().cpu().numpy()[0]

    def pi_loss(self, s: th.Tensor, a0: th.Tensor, p: th.Tensor):
        s = s.repeat([a0.shape[0], 1, 1])

        ts = np.exp(
            np.random.uniform(-2, 0, size=(a0.shape[0], a0.shape[1], 1, 1)) * np.log(10)
        )
        ts = th.tensor(ts, dtype=th.float32, device=self.device)
        at = cat_sample(apply_qt(a0, ts))

        s = s.view((a0.shape[0] * a0.shape[1], 4))
        at = at.view((a0.shape[0] * a0.shape[1], 3, 2))
        ts = ts.view((a0.shape[0] * a0.shape[1], 1))
        logits = self.calc_logits(s, at, ts)
        logits = logits.view(a0.shape)
        probs = th.softmax(logits, dim=-1)
        to_return = th.mean(th.mean(-th.log(probs) * a0, dim=(2, 3)) * p)
        return to_return
        # return nn.CrossEntropyLoss(p)(logits, a0)


class Buffer:
    def __init__(self, env: PrisonerEnv, agent: Agent, device):
        self.env = env
        self.agent = agent
        self.device = device

    def fill(self):
        self.all_states = []
        self.all_actions = []
        self.all_rewards = []
        self.all_dones = []

        for run_nb in range(100):
            state = self.env.reset()
            for t in range(10):
                action = self.agent.get_np_action(state)
                next_state, reward, done = self.env.step(action)

                self.all_states.append(state)
                self.all_actions.append(action)
                self.all_rewards.append(reward)
                self.all_dones.append(1 if done else 0)

                state = next_state
                if done:
                    break

        self.all_states = np.stack(self.all_states)
        self.all_actions = np.stack(self.all_actions)
        self.all_rewards = np.stack(self.all_rewards)
        self.all_dones = np.stack(self.all_dones)

        print(self.all_rewards)

    def sample(self):
        batch_size = 5
        ids = list(range(-1, len(self.all_actions) - 1))
        ids = np.random.permutation(ids)
        for batch_start in range(0, len(ids), batch_size):
            batch_end = batch_start + batch_size
            if batch_end <= len(ids):
                yield self.generate_batch(ids[batch_start:batch_end])

    def generate_batch(self, ids):
        return (
            th.tensor(self.all_states[ids], dtype=th.float32, device=self.device),
            th.tensor(self.all_actions[ids], dtype=th.float32, device=self.device),
            th.tensor(self.all_states[ids + 1], dtype=th.float32, device=self.device),
            th.tensor(self.all_rewards[ids], dtype=th.float32, device=self.device),
            th.tensor(self.all_dones[ids], dtype=th.float32, device=self.device),
        )


@dataclasses.dataclass
class DiffDqnConfig:
    env: PrisonerEnv
    agent: Agent
    old_agent: Agent
    device: str
    optimizer: th.optim.Optimizer
    wandb: bool = False


def start_diff_dqn(config: DiffDqnConfig):

    # create and fill buffer of trajs
    buffer = Buffer(config.env, config.agent, config.device)
    buffer.fill()

    print("buffer filled")

    # # training the q function
    # for epoch in range(100):
    #     for states, actions, next_states, rewards, done in buffer.sample():

    #         with th.no_grad():
    #             next_actions = config.old_agent.sample(next_states)
    #             next_q = config.old_agent.calc_q(next_states, next_actions)
    #             q_target = rewards + config.env.gamma * next_q * (1 - done)

    #         q_pred = config.agent.calc_q(states, actions)
    #         q_loss = th.mean(th.square(q_target - q_pred))

    #         q_loss.backward()
    #         config.optimizer.step()
    #         config.optimizer.zero_grad()

    #         if done[0].item() == 0:
    #             print(
    #                 actions[0, :, 0].detach().cpu().numpy(),
    #                 q_target[0].item(),
    #                 q_pred[0].item(),
    #             )

    #     print("updating target")
    #     config.old_agent.load_state_dict(config.agent.state_dict())

    # exit()

    for epoch in range(100):
        for states, actions, next_states, rewards, done in buffer.sample():
            # generate a bunch of possible actions
            with th.no_grad():
                all_next_actions = []
                all_next_q = []
                for i in range(10):
                    next_actions = config.old_agent.sample(next_states)
                    all_next_actions.append(next_actions)
                    all_next_q.append(
                        config.old_agent.calc_q(next_states, next_actions)
                    )

                all_next_actions = th.stack(all_next_actions)  # next_batch, batch, feat
                all_next_q = th.stack(all_next_q)

                count_actions(all_next_actions)

                mean_next_q = th.mean(all_next_q, dim=0)
                q_prob = th.softmax(all_next_q, dim=0)  # next_batch, batch

            q_target = rewards + config.env.gamma * mean_next_q * (1 - done)
            q_pred = config.agent.calc_q(states, actions)
            q_loss = th.mean(th.square(q_target - q_pred))
            # if done[0].item() == 0:
            #     print(
            #         actions[0, :, 0].detach().cpu().numpy(),
            #         q_target[0].item(),
            #         q_pred[0].item(),
            #     )

            pi_loss = config.agent.pi_loss(next_states, all_next_actions, q_prob)

            loss: th.Tensor = q_loss + pi_loss

            loss.backward()
            config.optimizer.step()
            config.optimizer.zero_grad()

            # test(config)

        print("updating target")
        config.old_agent.load_state_dict(config.agent.state_dict())


def count_actions(actions):
    actions = actions.detach().cpu().numpy()
    s = np.sum(actions[:, :, :, 0], axis=2)

    to_print = []
    for i in range(4):
        to_print.append(np.sum(s == i))
        # print("{}: {}".format(i, np.sum(s == i)), end)
    print(to_print)


def test(config):

    obs = np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, -1],
            [0, 1, 0, -2],
            [0, 1, 0, 1],
        ]
    )
    obs = th.tensor(obs, dtype=th.float32, device=config.device)
    act0 = [[0, 0, 0], [1, 1, 1]]
    act1 = [[1, 0, 0], [0, 1, 1]]
    act2 = [[1, 1, 0], [0, 0, 1]]
    act3 = [[1, 1, 1], [0, 0, 0]]
    act = np.array([act0, act1, act2, act3, act3, act3, act3, act3])
    act = th.tensor(act, dtype=th.float32, device=config.device)

    to_print = config.agent.calc_q(obs, act).detach().cpu().numpy()
    print(to_print[:4], to_print[4:])


def train(args):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    agent = Agent(device).to(device)
    old_agent = Agent(device).to(device)
    old_agent.load_state_dict(agent.state_dict())

    config = DiffDqnConfig(
        env=PrisonerEnv(),
        agent=agent,
        old_agent=old_agent,
        device=device,
        optimizer=th.optim.Adam(agent.parameters(), lr=3e-4),
    )

    start_diff_dqn(config)

    return config


config = train(42)


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--wandb",
#         action="store_true",
#         help="Whether to use wandb or not.",
#     )

#     args = parser.parse_args()

#     train(args)
