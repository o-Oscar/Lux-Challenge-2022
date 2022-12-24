import numpy as np
import torch as th

from utils import teams


def get_team_rollout(rollout, key, team):
    to_return = [x[team] for x in rollout[key]]
    return to_return


def map_robots_grid(robot_pos, next_robot_pos, new_value):
    to_return = new_value * 0
    for robot_id, next_pos in next_robot_pos.items():
        if robot_id in robot_pos:
            cur_pos = robot_pos[robot_id]
            to_return[cur_pos[0], cur_pos[1]] = new_value[next_pos[0], next_pos[1]]
    return to_return


class ReplayBuffer:
    def __init__(self, buffers=None):
        self.all_obs = []
        self.all_actions = []
        self.all_rewards = []
        self.all_masks = []
        self.all_unit_pos = []

        if buffers is not None:
            for buffer in buffers:
                self.all_obs.extend(buffer.all_obs)
                self.all_actions.extend(buffer.all_actions)
                self.all_rewards.extend(buffer.all_rewards)
                self.all_masks.extend(buffer.all_masks)
                self.all_unit_pos.extend(buffer.all_unit_pos)

    def split(self, ratio):
        """
        ratio : 0.9 for 90% of train
        """
        split_id = int(len(self.all_obs) * ratio)

        train_buffer = ReplayBuffer()
        test_buffer = ReplayBuffer()

        train_buffer.all_obs = self.all_obs[:split_id]
        train_buffer.all_actions = self.all_actions[:split_id]
        train_buffer.all_rewards = self.all_rewards[:split_id]
        train_buffer.all_masks = self.all_masks[:split_id]
        train_buffer.all_unit_pos = self.all_unit_pos[:split_id]

        test_buffer.all_obs = self.all_obs[split_id:]
        test_buffer.all_actions = self.all_actions[split_id:]
        test_buffer.all_rewards = self.all_rewards[split_id:]
        test_buffer.all_masks = self.all_masks[split_id:]
        test_buffer.all_unit_pos = self.all_unit_pos[split_id:]

        return train_buffer, test_buffer

    def expand(self, rollout):
        for team in teams:
            self.all_obs.append(get_team_rollout(rollout, "obs", team))
            self.all_actions.append(get_team_rollout(rollout, "actions", team))
            self.all_rewards.append(get_team_rollout(rollout, "rewards", team))
            self.all_masks.append(get_team_rollout(rollout, "masks", team))
            self.all_unit_pos.append(get_team_rollout(rollout, "unit_pos", team))

    def calc_q_target(self, agent, device, gamma):
        with th.no_grad():
            all_mean_q = []
            for obs in self.sample_targets(10, device):
                mean_q = agent.v_eval(obs)
                all_mean_q.append(mean_q.detach().cpu().numpy())
            all_mean_q = np.concatenate(all_mean_q, axis=0)

            self.all_q_target = []
            count = 0
            for i in range(len(self.all_actions)):
                cur_q_targets = []
                for t in range(len(self.all_actions[i])):
                    q_target = self.all_rewards[i][t]
                    if len(self.all_unit_pos[i]) < t - 1 and True:
                        q_target = q_target + gamma * map_robots_grid(
                            self.all_unit_pos[i][t],
                            self.all_unit_pos[i][t + 1],
                            all_mean_q[count + 1],
                        )
                    # if (
                    #     self.all_masks[i][t][0, 10, 12]
                    #     and self.all_actions[i][t][10, 12] == 0
                    # ):
                    #     print(q_target[10, 12])
                    cur_q_targets.append(q_target)
                    count += 1
                self.all_q_target.append(cur_q_targets)

    def sample_targets(self, batch_size, device):
        ids = []
        for i in range(len(self.all_actions)):
            ids = ids + [(i, t) for t in range(len(self.all_actions[i]))]

        # ids = np.random.permutation(ids) # Do not uncomment
        for batch_start in range(0, len(ids), batch_size):
            batch_end = batch_start + batch_size
            if batch_end <= len(ids):
                yield self.generate_target_batch(ids[batch_start:batch_end], device)

    def generate_target_batch(self, ids, device):
        obs = []
        for id in ids:
            obs.append(self.all_obs[id[0]][id[1]])
        obs = np.stack(obs, axis=0)
        obs = th.tensor(obs, device=device, dtype=th.float32)
        return obs

    def __len__(self):
        return sum([len(x) for x in self.all_actions])

    def sample(self, batch_size):
        ids = []
        for i in range(len(self.all_actions)):
            ids = ids + [(i, t) for t in range(len(self.all_actions[i]))]

        # ids = np.random.permutation(ids)
        for batch_start in range(0, len(ids), batch_size):
            batch_end = batch_start + batch_size
            if batch_end <= len(ids):
                yield self.generate_batch(ids[batch_start:batch_end])

    def reset(self):
        self.ids = []
        for i in range(len(self.all_actions)):
            self.ids.extend([(i, t) for t in range(len(self.all_actions[i]))])
        self.ids = np.random.permutation(self.ids)
        self.batch_start = 0

    def next(self, batch_size, device):
        bs = self.batch_start
        be = self.batch_start + batch_size
        be = min(len(self.ids), be)

        self.batch_start = be

        return self.generate_batch(self.ids[bs:be], device)

    def at_end(self, batch_size):
        return self.batch_start + batch_size >= len(self.ids)

    def generate_batch(self, ids, device):
        obs = []
        masks = []
        actions = []
        targets = []

        for id in ids:
            obs.append(self.all_obs[id[0]][id[1]])
            masks.append(self.all_masks[id[0]][id[1]])
            actions.append(self.all_actions[id[0]][id[1]])
            targets.append(self.all_q_target[id[0]][id[1]])

        obs = np.stack(obs, axis=0)
        masks = np.stack(masks, axis=0)
        actions = np.stack(actions, axis=0)
        targets = np.stack(targets, axis=0)

        obs = th.tensor(obs, device=device, dtype=th.float32)
        masks = th.tensor(masks, device=device, dtype=th.float32)
        actions = th.tensor(actions, device=device, dtype=th.float32)
        targets = th.tensor(targets, device=device, dtype=th.float32)

        return (
            obs,
            masks,
            actions,
            targets,
        )
