import argparse


def argument_parser():
    """
    This script transforms the searched changing points to cumulative ReLUs form.
    """
    parser = argparse.ArgumentParser(description="transform-results")
    parser.add_argument("--results-path", default="results/sa_search_ReSiLU2.txt", type=str, help="the path of the search results")
    parser.add_argument("--output-path", default="results/sa_best_ReSiLU2.txt", type=str, help="the path of the output")
    return parser


if __name__ == "__main__":
    args = argument_parser().parse_args()

    min_l2_dist = 1e9

    with open(args.results_path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        seperate_char = ','

        l2_dist = eval(line[:line.find(seperate_char)].strip())
        if l2_dist < min_l2_dist:
            min_l2_dist = l2_dist
            changing_points = eval(line[line.find(seperate_char) + len(seperate_char):].strip())

    print("min L2 distance:", min_l2_dist)
    print("best changing points:", changing_points)

    num_points = len(changing_points) // 2
    x_list = changing_points[:num_points]
    y_list = changing_points[num_points:]

    a_list = []
    c_list = []
    last_k = 0.
    for i in range(num_points):
        if i == num_points - 1:
            k = 1.
        else:
            k = (y_list[i + 1] - y_list[i]) / (x_list[i + 1] - x_list[i])
        a_list.append(k - last_k)
        c_list.append(x_list[i])
        last_k = k
    print("best ReSiLU2 a:", a_list)
    print("best ReSiLU2 c:", c_list)
    print("constraint condition 1:", sum(a_list))
    print("constraint condition 2:", sum((a * c for a, c in zip(a_list, c_list))))

    with open(args.output_path, 'w') as f:
        f.write("a: " + str(a_list) + "\n")
        f.write("c: " + str(c_list) + "\n")