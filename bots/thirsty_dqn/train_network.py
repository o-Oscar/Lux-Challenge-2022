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


def train_agent(args):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    action_handler = MoveActionHandler()
    obs_generator = CompleteObsGenerator()
    reward_generator = GlobalSurvivorRewardGenerator()

    env = Env(action_handler, obs_generator, reward_generator)

    # create the agent
    agent = Agent(env, device).to(device)
    target_agent = Agent(env, device).to(device)

    # load agent weights if need be
    if args.inp is not None:
        load_path: Path = Path("results/thirsty_dqn/agents") / args.inp
        target_agent.load_state_dict(th.load(load_path))

    # save the agent
    save_path: Path = Path("results/thirsty_dqn/agents") / args.out
    save_path.parent.mkdir(parents=True, exist_ok=True)
    th.save(agent.state_dict(), save_path)

    # return early if there is no buffer to train from
    if args.buffers is None or len(args.buffers) == 0:
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
    # phat TODO

    if args.wandb:
        wandb.init(project="lux_ai_dqn", name="dont got no name")

    # target_agent.load_state_dict(agent.state_dict())
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)

    start_time = time.time()

    q_loss_log = MeanLogger()
    v_loss_log = MeanLogger()

    batch_size = 64
    gamma = 0.99
    tau = 0.03
    polyak = 0.995

    train_buffer.calc_q_target(target_agent, device, gamma)
    train_buffer.reset()

    epoch = 0
    for update in range(1000):
        if train_buffer.at_end(batch_size):

            # logging
            to_log = {}
            to_log["main/q_loss"] = q_loss_log.value
            to_log["main/v_loss"] = v_loss_log.value
            epoch += 1
            print("Epoch {}, update {}".format(epoch, update))
            if not args.wandb:
                print(to_log)

            # saving the model
            th.save(agent.state_dict(), save_path)

            # reseting the buffer
            train_buffer.calc_q_target(target_agent, device, gamma)
            train_buffer.reset()
            q_loss_log.reset()
            v_loss_log.reset()

            # tau = np.exp(np.log(1e-2) * update / 300)
            # tau = 1

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
            # new_q2 = agent.q_eval(obs, new_actions, masks)
            # print(target_agent.state_dict()["q_network.0.weight"].data[0, 0, 0, 0])
            # exit()
            # print(th.sum(th.square(new_q - new_q2)))

        v_diff = new_q - agent.v_eval(obs)
        diff_weight = th.minimum(
            v_diff / tau, th.tensor(1, dtype=th.float32, device=device)
        ).detach()
        v_loss = (
            th.sum(robot_mask * th.square(v_diff) * th.exp(diff_weight)) / batch_size
        )
        # v_loss = th.sum(robot_mask * th.square(v_diff)) / batch_size
        v_coef = th.sum(robot_mask * th.exp(diff_weight).detach()) / th.sum(robot_mask)

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

        if (update + 1) % 100 == 0:
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

        if args.wandb:
            wandb.log(to_log)

        # with th.no_grad():
        #     for ap, tp in zip(agent.parameters(), target_agent.parameters()):
        #         tp = tp * polyak + ap * (1 - polyak)


"""
Usage :
python -m bots.thirsty_dqn.train_network --out 0000
python -m bots.thirsty_dqn.train_network --out 0001 --buffers 0000 --inp 0000 --wandb
python -m bots.thirsty_dqn.train_network --out 0002 --buffers 0000 0001 --inp 0001 --wandb
python -m bots.thirsty_dqn.train_network --out notargetupdate0 --buffers 0000 0001 --wandb
python -m bots.thirsty_dqn.train_network --inp notargetupdate0 --out notargetupdate1 --buffers 0000 0001 --wandb
python -m bots.thirsty_dqn.train_network --inp notargetupdate1 --out notargetupdate2 --buffers 0000 0001 --wandb
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
