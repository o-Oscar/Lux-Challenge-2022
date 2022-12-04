import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical


def create_data(batch_size, width, device):
    to_return = np.zeros((batch_size, 2, width, width))
    to_return[:, 1] = 1
    for i in range(batch_size):
        if False:
            randy = np.random.randint(2) + 1
            randx = randy
        else:
            randx = np.random.randint(5)
            randy = np.random.randint(5)
        to_return[i, 0, randx, randy] = 1
        to_return[i, 1, randx, randy] = 0

    return th.tensor(to_return, dtype=th.float32, device=device)


def create_dataset(width, device):
    batch_size = 25
    to_return = np.zeros((batch_size, 2, width, width))
    to_return[:, 1] = 1
    for i in range(batch_size):
        randx = i % 5
        randy = i // 5
        to_return[i, 0, randx, randy] = 1
        to_return[i, 1, randx, randy] = 0

    return th.tensor(to_return, dtype=th.float32, device=device)


def show_images(data, num_samples=20, cols=4, figsize=(15, 15), vmin=None, vmax=None):
    data = data.detach().cpu().numpy()
    """ Plots some samples from the dataset """
    plt.figure(figsize=figsize)
    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplot(num_samples // cols + 1, cols, i + 1)
        plt.imshow(img[0], vmin=vmin, vmax=vmax)
    plt.show()


def apply_qt(inp, t):
    e = th.exp(-t)
    f = (1 - e) / 2
    s = th.sum(inp, dim=1, keepdim=True)
    return e * inp + f * s


def sample(inp):
    probs = Categorical(probs=inp.permute(0, 2, 3, 1))
    return (
        nn.functional.one_hot(probs.sample(), num_classes=2)
        .permute(0, 3, 1, 2)
        .type(th.float32)
    )


def calc_logits(model, xs, ts, dim=1):
    inp = th.concat((xs, th.log(ts)), dim=dim)
    return model(inp)


def sample_traj(model, batch_size, device, num):
    all_t = np.logspace(-2, 0, num=num)

    cur_x = sample(th.ones(batch_size, 2, 5, 5).to(device) / 2)
    last_x = cur_x

    all_x = []
    all_t_inp = []
    # all_prob = []
    # all_new_x_prob = []

    # all_sum = []
    # all_xdt = []
    # all_probsdt = []

    all_x0 = []
    for i in range(2):
        x0 = np.zeros((1, 2, 5, 5))
        x0[0, i] = 1

        all_x0.append(th.tensor(x0, dtype=th.float32, device=device))

    for i in range(len(all_t) - 1, -1, -1):
        cur_x = last_x
        all_x.append(cur_x)

        if i == 0:
            dt = all_t[i]
        else:
            dt = all_t[i] - all_t[i - 1]

        t = all_t[i]
        # print(t)

        t_inp = th.zeros_like(cur_x[:, 0:1]) + t
        all_t_inp.append(t_inp)

        logits = calc_logits(model, cur_x, t_inp)
        probs = th.exp(logits) / th.sum(th.exp(logits), dim=1, keepdim=True)

        new_x_probs = th.zeros_like(probs)
        for x0 in all_x0:
            p_x0 = th.sum(probs * x0, dim=1, keepdims=True)
            sum = th.sum(x0 * apply_qt(cur_x, t_inp * 0 + t), dim=1, keepdim=True)
            new_x_probs = (
                new_x_probs
                + p_x0
                * (apply_qt(cur_x, t_inp * 0 + dt) * apply_qt(x0, t_inp * 0 + t - dt))
                / sum
            )
        last_x = sample(new_x_probs)

    #     all_x.append(cur_x)
    #     all_prob.append(probs)
    #     all_new_x_prob.append(new_x_probs)

    #     all_sum.append(sum)
    #     all_xdt.append(apply_qt(cur_x, t_inp * 0 + dt))
    #     all_probsdt.append(apply_qt(probs, t_inp * 0 + t - dt))

    # all_x = th.stack(all_x, dim=0)
    # all_prob = th.stack(all_prob, dim=0)
    # all_new_x_prob = th.stack(all_new_x_prob, dim=0)
    # all_sum = th.stack(all_sum, dim=0)
    # all_xdt = th.stack(all_xdt, dim=0)
    # all_probsdt = th.stack(all_probsdt, dim=0)

    all_x = th.concat(all_x, dim=0)
    all_t_inp = th.concat(all_t_inp, dim=0)
    last_x = th.tile(last_x, (len(all_t), 1, 1, 1))

    return (last_x, all_x, all_t_inp)


def compute_rewards(last_x):
    """
    last_x : (batch, channel, height, width)
    """
    return -th.square(th.sum(last_x[:, 0:1], dim=(2, 3), keepdim=True) - 1)


def calc_loss(rewards, last_x, logits):
    logits = logits - th.mean(logits, dim=1, keepdim=True)
    return -th.mean(rewards * last_x * logits)


# def show_plot(inp, )


def sample_traj_full(model, batch_size, device, num):
    all_t = np.logspace(-2, 0, num=num)

    cur_x = sample(th.ones(batch_size, 2, 5, 5).to(device) / 2)
    last_x = cur_x

    all_x = []
    all_t_inp = []
    all_prob = []
    all_new_x_prob = []

    all_sum = []
    all_xdt = []
    all_probsdt = []

    all_x0 = []
    for i in range(2):
        x0 = np.zeros((1, 2, 5, 5))
        x0[0, i] = 1

        all_x0.append(th.tensor(x0, dtype=th.float32, device=device))

    for i in range(len(all_t) - 1, -1, -1):
        cur_x = last_x
        all_x.append(cur_x)

        if i == 0:
            dt = all_t[i]
        else:
            dt = all_t[i] - all_t[i - 1]

        t = all_t[i]
        # print(t)

        t_inp = th.zeros_like(cur_x[:, 0:1]) + t
        all_t_inp.append(t_inp)

        logits = calc_logits(model, cur_x, t_inp)
        probs = th.exp(logits) / th.sum(th.exp(logits), dim=1, keepdim=True)

        new_x_probs = th.zeros_like(probs)
        for x0 in all_x0:
            p_x0 = th.sum(probs * x0, dim=1, keepdims=True)
            sum = th.sum(x0 * apply_qt(cur_x, t_inp * 0 + t), dim=1, keepdim=True)
            new_x_probs = (
                new_x_probs
                + p_x0
                * (apply_qt(cur_x, t_inp * 0 + dt) * apply_qt(x0, t_inp * 0 + t - dt))
                / sum
            )
        last_x = sample(new_x_probs)

        all_prob.append(probs)
        all_new_x_prob.append(new_x_probs)

        all_sum.append(sum)
        all_xdt.append(apply_qt(cur_x, t_inp * 0 + dt))
        all_probsdt.append(apply_qt(probs, t_inp * 0 + t - dt))

    # all_x = th.stack(all_x, dim=0)
    # all_prob = th.stack(all_prob, dim=0)
    # all_new_x_prob = th.stack(all_new_x_prob, dim=0)
    # all_sum = th.stack(all_sum, dim=0)
    # all_xdt = th.stack(all_xdt, dim=0)
    # all_probsdt = th.stack(all_probsdt, dim=0)

    all_x = th.stack(all_x, dim=0)
    all_prob = th.stack(all_prob, dim=0)
    all_new_x_prob = th.stack(all_new_x_prob, dim=0)
    all_sum = th.stack(all_sum, dim=0)
    all_xdt = th.stack(all_xdt, dim=0)
    all_probsdt = th.stack(all_probsdt, dim=0)

    all_t_inp = th.stack(all_t_inp, dim=0)
    last_x = th.tile(last_x, (len(all_t), 1, 1, 1))

    return (
        all_x,
        all_prob,
        all_new_x_prob,
        all_sum,
        all_xdt,
        all_probsdt,
        all_x,
        all_t_inp,
    )
