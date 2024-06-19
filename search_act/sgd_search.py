import math
import os
import argparse
import torch


def get_gelu_truncate_bounds(truncated_error):
    def gelu(x):
        y = 1 + math.erf(x / 2 ** 0.5)
        y *= x / 2
        return y

    def check_condition(x):
        condition = abs(x - gelu(x)) < 1
        return condition

    one_side_truncated_error = truncated_error / 2
    bound = math.sqrt(- math.log(one_side_truncated_error * 2) * 2)
    assert check_condition(bound), "condition must be met"
    return -bound, bound


def get_silu_truncate_bounds(truncated_error):
    def silu(x):
        y = x / (1 + math.exp(-x))
        return y

    def check_condition(x):
        condition_1 = abs(x - silu(x)) < 1
        condition_2 = math.exp(abs(x) / 2) - abs(x) - 1 > 0
        condition = condition_1 and condition_2
        return condition

    one_side_truncated_error = truncated_error / 2
    bound = - 2 * math.log(one_side_truncated_error)
    assert check_condition(bound), "condition must be met"
    return -bound, bound


class ApproxAct(torch.nn.Module):
    def __init__(self, bits: int, b_bounds: tuple = None): # type: ignore
        super().__init__()
        self.bounds = b_bounds or (- 100, 100)
        # head and tail elements are added purely for slope calculation
        self.x_list = torch.nn.Parameter(torch.empty((2 ** bits - 1 + 2), dtype=torch.double))
        self.y_list = torch.nn.Parameter(torch.empty((2 ** bits - 1 + 2), dtype=torch.double))

        torch.nn.init.uniform_(self.x_list, -4, 4)
        torch.nn.init.normal_(self.y_list, std=0.02)


    @torch.no_grad()
    def clamp(self):
        torch.clamp_(self.x_list.data, min=self.bounds[0], max=self.bounds[1])
        sorted, indices = torch.sort(self.x_list.data)
        self.x_list.data.copy_(sorted)
        self.x_list.data[0] = self.bounds[0] * 2
        self.x_list.data[-1] = self.bounds[1] * 2

        self.y_list.data[0] = 0.
        self.y_list.data[1] = 0.
        self.y_list.data[-2] = self.x_list.data[-2]
        self.y_list.data[-1] = self.x_list.data[-1]

    def get_weight_bias(self):
        self.clamp()
        slope = self.y_list.diff() / (self.x_list.diff() + 1e-8)
        weight = slope.diff()
        bias = self.x_list[1:-1]
        return weight, bias

    def forward(self, x: torch.Tensor):
        if x.size(-1) > 1:
            x = x.unsqueeze(-1)
            unsqueeze_flag = True
        else:
            unsqueeze_flag = False

        weights, bias = self.get_weight_bias()
        x = torch.relu(x - bias.unsqueeze(0)) * weights
        return x.sum(1, keepdim=(not unsqueeze_flag))


def argument_parser():
    """
    This script uses simulated annealing to search ReGELU2 and ReSiLU2.
    """
    parser = argparse.ArgumentParser(description="sa-search")
    parser.add_argument("--truncated-error", default=1e-8, type=float, help="tolerable error used to truncate the integral interval")
    parser.add_argument("--kbit", default=2, type=int, help="the k hyper-parameter in the approximate activation function")
    parser.add_argument("--repeats", default=10, type=int, help="the number of repeating the sgd searching")
    parser.add_argument("--num-steps", default=10000, type=int, help="the total iterations of sgd")
    parser.add_argument("--batch-size", default=1024, type=int, help="the batch size of one iteration")
    parser.add_argument("--output-dir", default="results", type=str, help="the output directory")
    return parser


if __name__ == "__main__":
    args = argument_parser().parse_args()
    device = 'cuda'
    truncated_error = args.truncated_error
    kbit = args.kbit
    repeats = args.repeats
    num_steps = args.num_steps
    batch_size = args.batch_size
    output_dir = args.output_dir

    silu_truncate_bounds = get_silu_truncate_bounds(truncated_error)
    silu_results_path = os.path.join(output_dir, f"sgd_search_ReSiLU{kbit}.txt")
    with open(silu_results_path, 'w') as f:
        f.write(f"l2_dist, var_list\n")
    for run in range(repeats):
        resiluk = ApproxAct(bits=kbit, b_bounds=silu_truncate_bounds).to(device)
        optimizer = torch.optim.SGD(resiluk.parameters(), lr=0.01, momentum=0.9)
        # optimizer = torch.optim.AdamW(resiluk.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

        for iter in range(num_steps):
            samples = torch.rand((batch_size,), dtype=torch.double, device=device) * (silu_truncate_bounds[1] - silu_truncate_bounds[0]) + silu_truncate_bounds[0]
            diff = resiluk(samples) - torch.nn.functional.silu(samples)
            loss = diff.pow(2).mean() * (silu_truncate_bounds[1] - silu_truncate_bounds[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (iter + 1) % 1000 == 0:
                print(f"\rsdg search ReSiLU{kbit} | repeat: {run+1}/{repeats} | iteration: {iter+1}/{num_steps} | L2 distance: {loss.item()}", end='', flush=True)

        with torch.no_grad():
            eval_samples = torch.rand((102400,), dtype=torch.double, device=device) * (silu_truncate_bounds[1] - silu_truncate_bounds[0]) + silu_truncate_bounds[0]
            diff = resiluk(samples) - torch.nn.functional.silu(samples)
            loss = diff.pow(2).mean() * (silu_truncate_bounds[1] - silu_truncate_bounds[0])

            resiluk.clamp()
            x_list = resiluk.x_list[1:-1].cpu().tolist()
            y_list = resiluk.y_list[1:-1].cpu().tolist()
            with open(silu_results_path, 'a') as f:
                f.write(f"{loss.item()}, {str(x_list + y_list)}\n")
    print(f"\nResults are written to {silu_results_path}")

    gelu_truncate_bounds = get_gelu_truncate_bounds(1e-10)
    gelu_results_path = os.path.join(output_dir, f"sgd_search_ReGELU{kbit}.txt")
    with open(gelu_results_path, 'w') as f:
        f.write(f"l2_dist, var_list\n")
    for run in range(repeats):
        regeluk = ApproxAct(bits=kbit, b_bounds=gelu_truncate_bounds).to(device)
        optimizer = torch.optim.SGD(regeluk.parameters(), lr=0.01, momentum=0.9)
        # optimizer = torch.optim.AdamW(regeluk.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

        for iter in range(num_steps):
            samples = torch.rand((batch_size,), dtype=torch.double, device=device) * (gelu_truncate_bounds[1] - gelu_truncate_bounds[0]) + gelu_truncate_bounds[0]
            diff = regeluk(samples) - torch.nn.functional.gelu(samples)
            loss = diff.pow(2).mean() * (gelu_truncate_bounds[1] - gelu_truncate_bounds[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (iter + 1) % 1000 == 0:
                print(f"\rsdg search ReGELU{kbit} | repeat: {run+1}/{repeats} | iteration: {iter+1}/{num_steps} | L2 distance: {loss.item()}", end='', flush=True)

        with torch.no_grad():
            eval_samples = torch.rand((102400,), dtype=torch.double, device=device) * (gelu_truncate_bounds[1] - gelu_truncate_bounds[0]) + gelu_truncate_bounds[0]
            diff = regeluk(samples) - torch.nn.functional.gelu(samples)
            loss = diff.pow(2).mean() * (gelu_truncate_bounds[1] - gelu_truncate_bounds[0])

            regeluk.clamp()
            x_list = regeluk.x_list[1:-1].cpu().tolist()
            y_list = regeluk.y_list[1:-1].cpu().tolist()
            with open(gelu_results_path, 'a') as f:
                f.write(f"{loss.item()}, {str(x_list + y_list)}\n")
    print(f"\nResults are written to {gelu_results_path}")