import argparse
from loguru import logger

import torch
from pprint import pprint

import code_dataset
from decoders.model.sdsat_model import SDSATModel
from decoders.sdsat import SDSATDecoderGreedy, SDSATDecoderNucleus
from decoders.autoregression import ARDecoder
from bench import Benchmarker


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for the model and dataset.")

    # Device configuration
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation.")

    # Model configuration
    parser.add_argument("--model_name", type=str, default="codellama", help="Name of the model.")
    parser.add_argument("--model_path", type=str, default="", help="Path to the model file.")
    parser.add_argument("--sa_tokens", type=str, default="32011,32012,32013,32014,32015", help="Semantic adaptive Tokens.")

    # Dataset configuration
    parser.add_argument("--data_name", type=str, default="", help="Name of the dataset.")
    parser.add_argument("--version", type=str, default="14", help="Version of the dataset.")
    parser.add_argument("--split", type=str, default="test", help="Split of the dataset to use.")
    parser.add_argument("--data_limit", type=float, default=1., help="The proportion of subset.")

    # Benchmark configuration
    parser.add_argument("--result_dir", type=str, default="./", help="Directory to save results.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")

    # Generate configuration
    parser.add_argument("--do_sample", action='store_true', help="Whether to sample during generation.")
    parser.add_argument("--temperature", type=float, default=1., help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for nucleus sampling.")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k for sampling.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate.")

    # Decoder configuration
    parser.add_argument("--use_cache", action='store_true', help="Whether to use cache in the decoder.")
    parser.add_argument("--k_list", type=str, default="0,5", help="Number of semantic adaptive tokens to use.")

    return parser.parse_args()


def load_tokenizer_model(cfg):
    logger.info("Loading tokenizer and model...")
    model = SDSATModel.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = model.get_tokenizer()
    return tokenizer, model


def load_dataset(tokenizer, cfg):
    logger.info(f"Loading {cfg.data_name} data...")
    if cfg.data_name == 'humaneval':
        datas = code_dataset.get_humaneval_data()
    elif cfg.data_name == 'multiple-infilling':
        datas = code_dataset.get_multiple_infilling_data(cfg.model_name, "SPM")
    else:
        raise ValueError(f"The dataset {cfg.data_name} is not supported")

    dataset = code_dataset.CodeDataset(cfg.data_name, datas, tokenizer, cfg.data_limit)
    return dataset


def load_decoders(cfg, tokenizer, model):
    logger.info("Loading decoders...")
    generate_config = {
        "do_sample": cfg.do_sample,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "max_new_tokens": cfg.max_new_tokens
    }
    ar_decoder = ARDecoder(
        tokenizer=tokenizer,
        model=model,
        gen_config=generate_config,
        use_cache=cfg.use_cache,
        device=cfg.device,
    )
    if not cfg.do_sample:
        sdsat_decoder = SDSATDecoderGreedy(
            tokenizer=tokenizer,
            model=model,
            sats=cfg.sa_tokens,
            gen_config=generate_config,
            use_cache=cfg.use_cache,
            device=cfg.device,
        )
    else:
        sdsat_decoder = SDSATDecoderNucleus(
            tokenizer=tokenizer,
            model=model,
            sats=cfg.sa_tokens,
            gen_config=generate_config,
            use_cache=cfg.use_cache,
            device=cfg.device,
        )
    decoders = {
        "ar_decoder": ar_decoder,
        "sdsat_decoder": sdsat_decoder
    }
    return decoders


def compute_benchmark(cfg, tokenizer, dataset, model):
    decoders = load_decoders(cfg, tokenizer, model)

    benchmarker = Benchmarker(
        dataset=dataset,
        decoders=decoders,
        sample_method="nucleus" if cfg.do_sample else "greedy",
        k_list=cfg.k_list,
        result_dir=cfg.result_dir,
        device=cfg.device,
        debug=cfg.debug,
    )
    benchmarker.compute_total_time()


def main():
    cfg = parse_args()
    logger.info("Experiment configuration:")
    pprint(vars(cfg))

    tokenizer, model = load_tokenizer_model(cfg)
    dataset = load_dataset(tokenizer, cfg)

    compute_benchmark(cfg, tokenizer, dataset, model)
    

if __name__ == "__main__":
    main()