import os
import time
import json

import itertools
import csv
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from loguru import logger

from decoders.utils.scorer import Scorer


def makedirs(path):
    if path.endswith((".tsv", ".csv", ".txt")):
        path = "/".join(path.split("/")[:-1])

    if not os.path.exists(path):
        os.makedirs(path)

def check_zero_division(a, b):
    return "na" if b == 0 else round(a / b, 3)


class Benchmarker(object):
    def __init__(
            self,
            dataset: Dataset,
            decoders,
            sample_method,
            k_list: str="0,1,2",
            result_dir: str = None,
            device: str = "cuda",
            debug: bool = False,
    ):
        self.dataset = dataset
        self.decoders = decoders
        self.k_list = [int(i) for i in k_list.split(",")]
        self.device = device
        self.debug = debug

        self.sample_method = sample_method
        self.dataloader = DataLoader(
            dataset, collate_fn=dataset.collate_fn, batch_size=1, shuffle=False
        )
        self.result_dir = result_dir
        self.exp_dir = self._retrieve_exp_dir()

    def _synchronize(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _retrieve_exp_dir(self):
        file_name = self._retrieve_file_name()

        exp_dir = os.path.join(self.result_dir, file_name)
        makedirs(exp_dir)

        return exp_dir

    def _retrieve_file_name(self):
        return f"{self.sample_method}_{self.device}_{self.dataset.name}"

    @staticmethod
    def _write_on_file(path, item, i):

        makedirs(path)
        with open(path, "a", encoding='utf-8') as file:
            writer = csv.writer(file, delimiter="\t")
            # write header
            if os.stat(path).st_size == 0:
                writer.writerow(["i", "item"])
            writer.writerow([i, item])

    def write_plot(self, num_data, scorers, best_algorithm):
        logger.info("Writing & plot...")

        def safe_mean(values):
            valid_values = [value for value in values if value is not None]
            return sum(valid_values) / len(valid_values) if valid_values else None

        def save_raw(scorer, save_path):
            raw_data = {
                "accept_rates": scorer.accept_rates,
                "sat": scorer.sat,
                "iters": scorer.iters,
                "max_new_tokens": scorer.max_new_tokens,
                "tokens_per_s": scorer.tokens_per_s,
                "loop_times": scorer.loop_times,
                "tot_mean_iter": scorer.tot_mean_iter,
                "tot_mean_time": scorer.tot_mean_time,
                "total_input": scorer.total_input,
                "total_output": scorer.total_output,
            }
            raw_data_str = json.dumps(raw_data, ensure_ascii=False)
            with open(save_path, 'a', encoding='utf-8') as f:
                f.write(raw_data_str + '\n')

        def plot_data(scorers, save_dir):
            import matplotlib.pyplot as plt
            metrics = { "sats": [], "mean_times": [], "tokens_per_s": [], "accept_rates": [], "loop_times": [] }

            for scorer in scorers.values():
                max_new_tokens = scorer.max_new_tokens
                metrics["mean_times"].append(scorer.tot_mean_time)
                metrics["sats"].append(scorer.sat)
                metrics["tokens_per_s"].append(safe_mean(scorer.tokens_per_s))
                metrics["accept_rates"].append(safe_mean(scorer.accept_rates))
                metrics["loop_times"].append(safe_mean(scorer.loop_times))

            plots_info = [
                ("Mean Inference Time", "mean_times", "Number of adaptive tokens", "Time (ms)"),
                ("Tokens per second", "tokens_per_s", "Number of adaptive tokens", "Tokens/s"),
                ("Accept Rate", "accept_rates", "Number of adaptive tokens", "Rate"),
                ("Loop Time", "loop_times", "Number of adaptive tokens", "Time (ms)")
            ]

            for title, metric_key, x_label, y_label in plots_info:
                plt.figure()
                plt.plot(metrics["sats"], metrics[metric_key], label=self.dataset.name, marker='*', linestyle='--')
                plt.legend()
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.title(f'{title} (Dataset: {self.dataset.name})')
                plt.savefig(os.path.join(save_dir, f'{metric_key}.png'))

        raw_data_path = f"{self.exp_dir}/raw_data_{self.dataset.name}.json"
        if os.path.exists(raw_data_path):
            os.remove(raw_data_path)
        for name, scorer in scorers.items():
            save_raw(scorer, raw_data_path)
        plot_data(scorers, self.exp_dir)
        logger.info(f"Check {self.exp_dir} for results")

    def write_inline(self, i: int, scorers, best_alg):

        for name, scorer in scorers.items():
            # Write times
            path = os.path.join(self.exp_dir, name, f"{name}.tsv")
            self._write_on_file(path, scorer.current_time, i)
            # Write iterations
            path = os.path.join(self.exp_dir, name, f"iter_{name}.tsv")
            self._write_on_file(path, scorer.current_iter, i)
            # Write output sequences
            path = os.path.join(self.exp_dir, name, f"out_{name}.tsv")
            self._write_on_file(path, scorer.current_pred, i)

        # Write mean
        path = os.path.join(self.exp_dir, "meanvar.tsv")
        makedirs(path)
        with open(path, "a") as file:
            writer = csv.writer(file, delimiter="\t")

            # Write header
            if os.stat(path).st_size == 0:
                header = ["#sentence"] + [f"mean_{name}" for name in scorers] + ["best_alg"]
                writer.writerow(header)

            row = [i] + [scorer.current_time for scorer in scorers.values()] + [best_alg]
            writer.writerow(row)

    @staticmethod
    def _compute_best_algorithm(scorers):
        best = scorers.get(min(scorers, key=lambda x: scorers[x].current_time)).name
        return best

    def compute_total_time(self):
        i = 0
        scorers = {f'sat_{sat}': Scorer(f'sat_{sat}', sat) for sat in self.k_list}
        best_algorithms = {f'sat_{sat}': 0 for sat in self.k_list}
        for _, decoder in self.decoders.items():
            decoder.initialize()
        # self.decoders["sdsat_decoder"].initialize()

        pbar = tqdm(self.dataloader, desc="Computing Benchmark...")
        loop_limit = 3 if self.debug else len(pbar)
        
        for x in itertools.islice(pbar, loop_limit):
            target_len = None
            input_ids = x["source"]["input_ids"].to(self.device)
            attention_mask = x["source"]["attention_mask"].to(self.device)

            tgt_text = x['target']['sentences']

            for name, scorer in scorers.items():
                sat = scorer.sat
                decoder = self.decoders["ar_decoder"] if sat == 0 else self.decoders["sdsat_decoder"]
                # kwargs = decoder.compute_decode_kwargs(input_ids, attention_mask)

                self._synchronize()
                start = time.perf_counter()
                if name == "sat_0":
                    output_ids, iters, avg_loop_time, token_per_s = decoder.decode(input_ids, attention_mask, sat)
                    target_len = iters
                    accept_rate = 1
                else:
                    assert target_len is not None
                    output_ids, iters, accept_rate, avg_loop_time, token_per_s = decoder.decode(input_ids, attention_mask, sat, target_len)

                self._synchronize()
                end = time.perf_counter()
                sample_time = end - start

                # init_tensor = decoder.kwargs["init_tensor"] if "init_tensor" in decoder.kwargs else ""

                output_strs = self.dataset.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )[0]

                scorers[name].update_metrics(sample_time, input_ids, iters, accept_rate, avg_loop_time, \
                    token_per_s, output_ids, output_strs, tgt_text[0], decoder.generation_config['max_new_tokens'])

            best_alg = self._compute_best_algorithm(scorers)
            best_algorithms[best_alg] += 1

            # Update tqdm bar
            pbar.update(1)

            self.write_inline(i, scorers, best_alg)
            i += 1

        self.write_plot(i, scorers, best_algorithms)
