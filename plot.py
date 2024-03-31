import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser("plot args")
    parser.add_argument("--data_path", required=True, help="data path")
    parser.add_argument("--output_path", required=True, default=None)
    return parser.parse_args()


def draw(datas, save_dir):
    colors = ['c', 'orange', 'g', 'r', 'c']
    linestyles = ['-', '-', '-.', '--', ':']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def plot_data(data, xlabel, ylabel, title, filename, digits=2):
        plt.figure()
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
        for idx, (name, (sats, means, ses)) in enumerate(data.items()):
            # choose color and linestyle
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            plt.plot(sats, means, label=name, marker='*', color=color, linestyle=linestyle)
            # add text on points
            format_str = f'{{:.{digits}f}}'
            for x, y in zip(sats, means):
                if y is not None:
                    plt.text(x, y, format_str.format(y), color=color, fontsize=8)

            alpha_value = 0.15  # for lighter fill
            plt.fill_between(sats, [m - 1.96 * s for m, s in zip(means, ses)],
                             [m + 1.96 * s for m, s in zip(means, ses)], 
                             color=color,
                             alpha=alpha_value,
                             linestyle=linestyle)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename))

    def safe_mean(values):
        valid_values = [value for value in values if value is not None]
        return sum(valid_values) / len(valid_values) if valid_values else None

    def safe_mean_stderr(values_dict):
        xs, mean_values, std_errors = [], [], []
        for key, value in values_dict.items():
            if not None in value:
                mean_values.append(np.mean(value))
                std_errors.append(np.std(value) / np.sqrt(len(value)))
                xs.append(key)
        return xs, mean_values, std_errors

    iters_data = {}
    mean_times_data = {}
    accept_rates_data = {}
    loop_times_data = {}
    tokens_per_s_data = {}
    for name, multi_scorers in sorted(datas.items(), key=lambda x: x[0]):
        iters, tokens_per_s, mean_times, accept_rates, loop_times, sats = [], [], [], [], [], []
        max_new_tokens = multi_scorers[0][0]['max_new_tokens']
        sats = defaultdict(list)
        mean_times = defaultdict(list)
        iters = defaultdict(list)
        tokens_per_s = defaultdict(list)
        accept_rates = defaultdict(list)
        loop_times = defaultdict(list)
        for scorers in multi_scorers:
            for scorer in scorers:
                t = scorer['sat']
                sats[t].append(t)
                mean_times[t].append(scorer['tot_mean_time'])
                iters[t].append(scorer['tot_mean_iter'])

                tokens_per_s[t].append(safe_mean(scorer['tokens_per_s']))
                accept_rates[t].append(safe_mean(scorer['accept_rates']))
                loop_times[t].append(safe_mean(scorer['loop_times']))

        xs = [i[0] for i in sats.values()]
        mean_times_data[name] = safe_mean_stderr(mean_times)
        accept_rates_data[name] = safe_mean_stderr(accept_rates)
        loop_times_data[name] = safe_mean_stderr(loop_times)
        tokens_per_s_data[name] = safe_mean_stderr(tokens_per_s)

    # ploting
    plot_data(mean_times_data, 'Number of adaptive tokens (k)', 'Time (ms)', 'Mean Inference Time', 'mean_time.png')
    plot_data(accept_rates_data, 'Number of adaptive tokens (k)', 'Accept Rate', 'Accept Rate', 'accept_rate.png')
    plot_data(loop_times_data, 'Number of adaptive tokens (k)', 'Time (ms)', 'Inference Latency', 'loop_time.png')
    plot_data(tokens_per_s_data, 'Number of adaptive tokens (k)', 'Tokens/s', 'Tokens Per Second', 'token_per_s.png')



def run():
    args = parse_args()
    datas = defaultdict(list)
    for root, dirs, files in os.walk(args.data_path):
        for file in files:
            if not file.startswith("raw_data"): continue
            data_name = file.split('.')[0].split('_')[-1]
            fileP = os.path.join(root, file)
            with open(fileP, 'r') as f:
                lines = f.readlines()
                data = [json.loads(line) for line in lines]
            datas[data_name].append(data)
    draw(datas, args.output_path)
    logger.info(f"Success Plotting. Check {args.output_path} for results.")


if __name__ == '__main__':
    run()