import os
import argparse
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from scipy import special


def argument_parser():
    """
    This script plots the searched results in cumulative ReLUs form.
    """
    parser = argparse.ArgumentParser(description="plot-results")
    parser.add_argument("--target", default="silu", type=str, help="the target function")
    parser.add_argument("--results-path", default="results/sa_best_ReSiLU2.txt", type=str, help="the path of the search results")
    return parser


def gelu(x):
    y = 1 + special.erf(x / 2 ** 0.5)
    y *= x / 2
    return y


def dgelu(x):
    y = 1 + special.erf(x / 2 ** 0.5)
    y /= 2
    y += x * np.exp(- x ** 2 / 2) / (2 * np.pi) ** 0.5
    return y


def silu(x):
    y = x / (1 + np.exp(-x))
    return y


def dsilu(x):
    e_x = np.exp(-x)
    e_x_1 = e_x + 1
    y = (e_x_1 + x * e_x) / e_x_1 ** 2
    return y


def relu(x):
    y = np.abs(x) / 2 + x / 2
    return y


def approx_act(x, a, c):
    if x.shape[-1] > 1:
        x = x[:, None]
        unsqueeze_flag = True
    else:
        unsqueeze_flag = False
    y = relu(x - c[None, :])
    y = (y * a).sum(1, keepdims=not unsqueeze_flag)
    return y

def dapprox_act(x, a, c):
    if x.shape[-1] > 1:
        x = x[:, None]
        unsqueeze_flag = True
    else:
        unsqueeze_flag = False
    y = (x - c[None, :]) > 0.
    y = (y * a).sum(1, keepdims=not unsqueeze_flag)
    return y


if __name__ == '__main__':
    args = argument_parser().parse_args()

    target_act = args.target.lower()
    assert target_act in ['gelu', 'silu'], "this script only supports gelu and silu"
    if target_act == 'gelu':
        target_f = gelu
        target_d = dgelu
        target_act = 'GELU'
    elif target_act == 'silu':
        target_f = silu
        target_d = dsilu
        target_act = 'SiLU'

    with open(args.results_path, 'r') as f:
        lines = f.readlines()

    a_line = lines[0].strip()
    a_key_char = 'a:'
    a_list = eval(a_line[a_line.find(a_key_char) + len(a_key_char):].strip())
    a_array = np.array(a_list)

    c_line = lines[1].strip()
    c_key_char = 'c:'
    c_list = eval(c_line[c_line.find(c_key_char) + len(c_key_char):].strip())
    c_array = np.array(c_list)

    kbit = int(np.log2(len(a_list) + 1))

    font_size = 13
    label_size = 11.5
    lw = 2
    font = fm.FontProperties(size=font_size)

    n = 10000
    left_bound = -8
    right_bound = 8
    x = np.arange(n) / n * (right_bound - left_bound) + left_bound
    y_target = target_f(x)
    d_target = target_d(x)

    y_approx = approx_act(x, a_array, c_array)
    d_approx = dapprox_act(x, a_array, c_array)

    plt.plot(x, y_target, label=f'{target_act}', linewidth=lw)
    plt.plot(x, y_approx, label=f'{target_act}_{kbit}bit', linewidth=lw)

    plt.plot(x, d_target, label=f'd{target_act}', linewidth=lw)
    plt.plot(x, d_approx, label=f'd{target_act}_{kbit}bit', linewidth=lw)


    plt.xlabel('Input', fontproperties=font)
    plt.ylabel('Output', fontproperties=font)

    plt.xlim(left_bound, right_bound)
    plt.ylim(-0.5, right_bound)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.yticks(range(-1, right_bound + 1))
    plt.legend(shadow=False, fontsize=font_size, loc=0)
    plt.tick_params(labelsize=label_size)

    output_path = args.results_path.replace(".txt", "_plot.png")
    plt.savefig(output_path, bbox_inches='tight')
