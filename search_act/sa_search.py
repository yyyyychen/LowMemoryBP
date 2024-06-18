import math
import os
import argparse
from scipy import integrate
from functools import partial
import random
import numpy as np


def gelu(x):
    y = 1 + math.erf(x / 2 ** 0.5)
    y *= x / 2
    return y

def dgelu(x):
    y = 1 + math.erf(x / 2 ** 0.5)
    y /= 2
    y += x * math.exp(- x ** 2 / 2) / (2 * math.pi) ** 0.5
    return y

def silu(x):
    y = x / (1 + math.exp(-x))
    return y

def dsilu(x):
    e_x = math.exp(-x)
    e_x_1 = e_x + 1
    y = (e_x_1 + x * e_x) / e_x_1 ** 2
    return y


def segment_deviation(func, a_point, b_point):
    if b_point[0] - a_point[0] < 1e-8:
        d_2 = 0
    else:
        slope = (b_point[1] - a_point[1]) / (b_point[0] - a_point[0])
        bias = a_point[1] - slope * a_point[0]
        d_2, error = integrate.quad(lambda x: (func(x) - slope * x - bias) ** 2, a_point[0], b_point[0], epsabs=1e-10, limit=1000)
    return d_2


def linear_deviation(func, a_point, b_point, var_list):
    """
    func: univariate function
    a: the left bound
    b: the right bound
    var_list = x_list + y_list: the changing points
    """

    n = int(len(var_list) // 2)
    x_list, y_list = var_list[:n], var_list[n:]

    v = 0
    left_point = a_point
    right_point = (x_list[0], y_list[0])
    v += segment_deviation(func, left_point, right_point)
    for i in range(len(x_list) - 1):
        left_point = (x_list[i], y_list[i])
        right_point = (x_list[i + 1], y_list[i + 1])
        v += segment_deviation(func, left_point, right_point)
    left_point = (x_list[-1], y_list[-1])
    right_point = b_point
    v += segment_deviation(func, left_point, right_point)
    return v


def normal_random_walk(var_list, a, b, sigma=5):
    num_points = int(len(var_list) // 2)
    x_list = var_list[:num_points]
    y_list = var_list[num_points:]

    new_x_list = []
    for x in x_list:
        x += random.normalvariate(0, sigma)
        if x < a:
            x = a
        elif x > b:
            x = b
        new_x_list.append(x)
    new_x_list = sorted(new_x_list)

    new_y_list = []
    for y in y_list[1:-1]:
        y += random.normalvariate(0, sigma)
        new_y_list.append(y)
    new_y_list = [0,] + new_y_list + [new_x_list[-1],]

    new_var_list = new_x_list + new_y_list
    return new_var_list


class SimulatedAnnealing:
    def __init__(self, energe_func, walk_func, init_var_list, init_temp=1, init_sigma=1, num_fixed_var=0, max_step=10000):
        self.energe_func = energe_func
        self.walk_func = walk_func
        self.var_list = init_var_list
        self.num_fixed_var = num_fixed_var
        self.energe = self.energe_func(var_list=init_var_list)

        self.optimal_energe = self.energe
        self.optimal_var_list = self.var_list

        self.init_temp = init_temp
        self.init_sigma = init_sigma
        self.k = 0
        self.max_step = max_step
        self.last_step_trans = True
        self.stop_steps = 0

    def walk(self, sigma):
        var_list = self.var_list[:self.num_fixed_var] + self.walk_func(self.var_list[self.num_fixed_var:], sigma=sigma)
        return var_list

    def determine(self, energe_new, temp):
        if energe_new < self.energe:
            return True
        else:
            p = math.exp(-(energe_new - self.energe) / temp)
            if random.uniform(0, 1) < p:
                return True
            else:
                return False

    def step(self):
        new_var_list = self.walk(sigma=self.init_sigma * (1 - self.k / self.max_step))
        # new_var_list = self.walk(sigma=self.init_sigma / (self.k +  1))
        new_energe = self.energe_func(var_list=new_var_list)
        # if self.determine(new_energe, self.init_temp * (1 - self.k / self.max_step)):
        if self.determine(new_energe, self.init_temp / (self.k + 1)):
            self.var_list = new_var_list
            self.energe = new_energe
            self.last_step_trans = True
            self.stop_steps = 0
            if new_energe < self.optimal_energe:
                self.optimal_energe = new_energe
                self.optimal_var_list = new_var_list
        else:
            self.last_step_trans = False
            self.stop_steps += 1
        self.k += 1


def apply_1d(func, array_1d):
    out = np.apply_along_axis(func, arr=array_1d[:, None], axis=1)[:, 0]
    return out


def get_gelu_truncate_bounds(truncated_error):
    def check_condition(x):
        condition = abs(x - gelu(x)) < 1
        return condition
    one_side_truncated_error = truncated_error / 2
    bound = math.sqrt(- math.log(one_side_truncated_error * 2) * 2)
    assert check_condition(bound), "condition must be met"
    return -bound, bound


def get_dgelu_truncate_bounds(truncated_error):
    def check_condition(x):
        condition = abs(x) - math.exp(x ** 2 / 4) < 0
        return condition
    one_side_truncated_error = truncated_error / 2
    bound = math.sqrt(-4 * math.log(math.sqrt(2 * math.pi) * one_side_truncated_error))
    assert check_condition(bound), "condition |x| - exp(-x^2/4) < 0 must be met"
    return -bound, bound


def get_silu_truncate_bounds(truncated_error):
    def check_condition(x):
        condition_1 = abs(x - silu(x)) < 1
        condition_2 = math.exp(abs(x) / 2) - abs(x) - 1 > 0
        condition = condition_1 and condition_2
        return condition
    one_side_truncated_error = truncated_error / 2
    bound = - 2 * math.log(one_side_truncated_error)
    assert check_condition(bound), "condition e^{|x|/2} - |x| - 1 > 0 must be met"
    return -bound, bound


def get_dsilu_truncate_bounds(truncated_error):
    def check_condition(x):
        condition = abs(x) - 2 * math.log(abs(x)) > 0
        return condition
    one_side_truncated_error = truncated_error / 2
    bound = - 2 * math.log(one_side_truncated_error / 2)
    assert check_condition(bound), "condition |x| - 2 * ln(|x|) > 0 must be met"
    return -bound, bound


def argument_parser():
    """
    This script uses simulated annealing to search ReGELU2 and ReSiLU2.
    """
    parser = argparse.ArgumentParser(description="sa-search")
    parser.add_argument("--truncated-error", default=1e-8, type=float, help="tolerable error used to truncate the integral interval")
    parser.add_argument("--repeats", default=10, type=int, help="the number of repeating the simulated annealing searching")
    parser.add_argument("--init-sigma", default=1., type=float, help="initial standard deviation used in initializing paramemters and the random walk")
    parser.add_argument("--num-steps", default=1000000, type=int, help="the total steps of the random walk")
    parser.add_argument("--output-dir", default="results", type=str, help="the output directory")
    return parser


if __name__ == '__main__':
    args = argument_parser().parse_args()
    truncated_error = args.truncated_error
    repeats = args.repeats
    init_sigma = args.init_sigma
    num_steps = args.num_steps

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    silu_a, silu_b = get_silu_truncate_bounds(truncated_error)
    silu_a = (silu_a, 0)
    silu_b = (silu_b, silu_b)

    gelu_a, gelu_b = get_gelu_truncate_bounds(truncated_error)
    gelu_a = (gelu_a, 0)
    gelu_b = (gelu_b, gelu_b)


    def build_deviation_optimizer(func, init_var_list, a_point, b_point, init_sigma=10, init_temp=100, max_step=10000):
        deviation_energe = partial(linear_deviation, func=func, a_point=a_point, b_point=b_point)
        bounded_normal_walk = partial(normal_random_walk, a=a_point[0], b=b_point[0])
        optimizer = SimulatedAnnealing(energe_func=deviation_energe, walk_func=bounded_normal_walk, init_var_list=init_var_list, init_temp=init_temp, init_sigma=init_sigma, max_step=max_step)
        return optimizer


    bits = 2
    silu_results_path = os.path.join(output_dir, f"sa_search_ReSiLU{bits}.txt")
    with open(silu_results_path, 'w') as f:
        f.write(f"energe, var_list\n")
    for run in range(repeats):
        init_x_list = sorted([random.uniform(silu_a[0], silu_b[0]) for _ in range(2**bits - 1)])
        init_y_list = [0,] + [random.normalvariate(0, init_sigma) for _ in range(2**bits - 3)] + [init_x_list[-1],]
        init_var_list = init_x_list + init_y_list
        print(init_var_list)
        for restart in range(bits):
            optimizer = build_deviation_optimizer(silu, init_var_list, silu_a, silu_b, init_sigma=init_sigma, max_step=num_steps)
            for step in range(num_steps):
                optimizer.step()
                print(optimizer.energe, optimizer.var_list)
            init_var_list = optimizer.optimal_var_list
        with open(silu_results_path, 'a') as f:
            f.write(f"{optimizer.optimal_energe}, {str(optimizer.optimal_var_list)}\n")


    gelu_results_path = os.path.join(output_dir, f"sa_search_ReGELU{bits}.txt")
    with open(gelu_results_path, 'w') as f:
        f.write(f"energe, var_list\n")
    for run in range(repeats):
        init_x_list = sorted([random.uniform(gelu_a[0], gelu_b[0]) for _ in range(2**bits - 1)])
        init_y_list = [0,] + [random.normalvariate(0, init_sigma) for _ in range(2**bits - 3)] + [init_x_list[-1],]
        init_var_list = init_x_list + init_y_list
        for restart in range(bits):
            optimizer = build_deviation_optimizer(gelu, init_var_list, gelu_a, gelu_b, init_sigma=init_sigma, max_step=num_steps)
            for step in range(num_steps):
                optimizer.step()
                print(optimizer.energe, optimizer.var_list)
            init_var_list = optimizer.optimal_var_list
        with open(gelu_results_path, 'a') as f:
            f.write(f"{optimizer.optimal_energe}, {str(optimizer.optimal_var_list)}\n")