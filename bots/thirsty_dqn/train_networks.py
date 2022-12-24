import argparse
import pickle
import time
from pathlib import Path

import torch as th
import torch.optim as optim

import wandb
from bots.thirsty_dqn.agent import Agent
from learning.dqn_buffer import ReplayBuffer
from utils.action.move import MoveActionHandler
from utils.env import Env
from utils.obs.complete import CompleteObsGenerator
from utils.reward.survivor import GlobalSurvivorRewardGenerator


class MeanLogger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.sum = 0

    def update(self, value, n=1):
        self.n += n
        self.sum += value * n

    @property
    def value(self):
        return self.sum / self.n


class EarlyStopper:
    def __init__(self, max_not_best):
        self.max_not_best = max_not_best
        self.n_not_best = 0
        self.best_test_loss = None

    def update(self, test_loss):
        if self.best_test_loss is None:
            self.best_test_loss = test_loss
            return

        if self.best_test_loss > test_loss:
            self.best_test_loss = test_loss
        else:
            self.n_not_best += 1

        return self.is_done()

    def is_done(self):
        return self.n_not_best > self.max_not_best


def get_save_path(model_name, model_number):
    model_full_name = "{}_{:04d}".format(model_name, model_number)
    save_path: Path = Path("results/thirsty_dqn/agents") / model_full_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


def train_agent(args):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    action_handler = MoveActionHandler()
    obs_generator = CompleteObsGenerator()
    reward_generator = GlobalSurvivorRewardGenerator()

    env = Env(action_handler, obs_generator, reward_generator)

    # create the agent
    agent = Agent(env, device).to(device)
    target_agent = Agent(env, device).to(device)

    # load target agent weights if need be
    if args.inp is not None:
        load_path: Path = Path("results/thirsty_dqn/agents") / args.inp
        target_agent.load_state_dict(th.load(load_path))

    # save the agent
    save_path = get_save_path(args.out, 0)
    th.save(agent.state_dict(), save_path)

    # return early if there is no buffer to train from
    if len(args.buffers) == 0:
        print("No buffer. Returning early.")
        return

    # load and concatenate the buffers
    buffers = []
    for buffer_name in args.buffers:
        buffer_path: Path = Path("results/thirsty_dqn/buffers") / buffer_name
        with open(buffer_path, "rb") as f:
            buffers.append(pickle.load(f))
    buffer = ReplayBuffer(buffers)
    train_buffer, test_buffer = buffer.split(0.9)

    # train the agent on the buffer data

    if args.wandb:
        wandb.init(project="lux_ai_dqn", name="dont got no name")

    for model_nb in range(args.n):

        train_cur_agent(
            agent, target_agent, train_buffer, test_buffer, device, args, save_path
        )

        print("--- Bootstraping target agent ---")

        target_agent.load_state_dict(agent.state_dict())
        agent = Agent(env, device).to(device)

        save_path = get_save_path(args.out, model_nb)
        th.save(agent.state_dict(), save_path)


def train_cur_agent(
    agent, target_agent, train_buffer, test_buffer, device, args, save_path
):
    # target_agent.load_state_dict(agent.state_dict())
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)

    q_loss_log = MeanLogger()
    v_loss_log = MeanLogger()
    early_stopper = EarlyStopper(1)

    batch_size = 64
    gamma = 0.99
    tau = 0.03

    train_buffer.calc_q_target(target_agent, device, gamma)
    train_buffer.reset()

    epoch = 0
    for update in range(1000):
        if train_buffer.at_end(batch_size):
            print("Epoch {}, update {}".format(epoch, update))
            epoch += 1

            # cli logging
            to_print = {}
            to_print["main/q_loss"] = q_loss_log.value
            to_print["main/v_loss"] = v_loss_log.value
            if not args.wandb:
                print(to_print)

            # reseting the buffer
            train_buffer.reset()
            q_loss_log.reset()
            v_loss_log.reset()

        obs, masks, actions, q_target = train_buffer.next(batch_size, device)
        robot_mask = th.max(masks, dim=1).values

        q_pred = agent.q_eval(obs, actions, masks)
        q_loss = th.sum(robot_mask * th.square(q_pred - q_target)) / batch_size

        with th.no_grad():
            new_actions = (
                th.randint(
                    0, 5, size=(batch_size, 48, 48), dtype=th.long, device=device
                )
                * 0
            )
            new_q = target_agent.q_eval(obs, new_actions, masks).detach()

        v_diff = new_q - agent.v_eval(obs)
        diff_weight = th.minimum(
            v_diff / tau, th.tensor(1, dtype=th.float32, device=device)
        ).detach()
        v_loss = (
            th.sum(robot_mask * th.square(v_diff) * th.exp(diff_weight)) / batch_size
        )
        v_coef = th.sum(robot_mask * th.exp(diff_weight)) / th.sum(robot_mask)

        loss = q_loss + v_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        q_loss_log.update(q_loss.item())
        v_loss_log.update(v_loss.item())

        # wandb logging
        to_log = {}
        to_log["main/q_loss"] = q_loss.item()
        to_log["main/v_loss"] = v_loss.item()
        to_log["main/v_coef"] = v_coef.item()

        if (update + 1) % 30 == 0:
            # test computation
            test_buffer.calc_q_target(target_agent, device, gamma)
            test_buffer.reset()
            test_q_loss_log = MeanLogger()
            while not test_buffer.at_end(batch_size):
                obs, masks, actions, q_target = test_buffer.next(batch_size, device)
                robot_mask = th.max(masks, dim=1).values

                q_pred = agent.q_eval(obs, actions, masks)
                q_loss = th.sum(robot_mask * th.square(q_pred - q_target)) / batch_size
                test_q_loss_log.update(q_loss.item())
            to_log["test/q_loss"] = test_q_loss_log.value

            print(test_q_loss_log.value)
            early_stopper.update(test_q_loss_log.value)
            # changing the model
            if early_stopper.is_done():
                return

            # saving the model
            th.save(agent.state_dict(), save_path)

        if args.wandb:
            wandb.log(to_log)


"""
Usage :
python -m bots.thirsty_dqn.train_networks --out a --buffers 0000 -n 10 --wandb
python -m bots.thirsty_dqn.train_networks --out a --buffers 0000 --inp 0000 -n 10 --wandb
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--buffers",
        nargs="*",
        help="Buffer files to use for rollout data.",
    )
    parser.add_argument(
        "--inp",
        type=str,
        default=None,
        help="Input checkpoint file for the network.",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Out save file for the network.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to use wandb or not.",
    )
    parser.add_argument(
        "-n",
        type=int,
        help="Number of .",
    )

    args = parser.parse_args()

    train_agent(args)

"""
TODO : 
trouver un moyen d'arrêter automatiquement l'entraînement quand la test loss ne descend plus
rebooter l'entraînement en posant target = current et current = default init -> faire ça pluieurs fois et s'arranger pour pouvoir tester les performances à chaque fois
-> voir si on peut répéter la procédure à l'infini et avoir des performances qui grimpent toujours (ou en tout cas qui ne dégringollent pas)
(recommencer la récolte de dataset / entrainement de réseaux depuis le début)
essayer de trouver d'autres méthodes de régularisation (l1, l2 ?)
"""
