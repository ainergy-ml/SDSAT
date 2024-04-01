# SDSAT
The official implementation of "SDSAT: Accelerating LLM Inference through Speculative Decoding with Semantic Adaptive Tokens"

📖 [paper](https://arxiv.org/abs/2403.18647)

Including the decoding method of both greedy search and nucleus sampling.


## Introduction

SDSAT is an acceleration scheme for large language models (LLMs) through Speculative Decoding with Semantic Adaptive Tokens (SDSAT). The primary objective of this design is to enhance the LLM model’s ability to generate draft tokens more accurately without compromising the model’s accuracy. The core strategies involve: 
1. Fine-tune the model by incorpo-rating semantic adaptive tokens that possess flexible decoding capabilities without changing its structure, allowing them to generate high-quality draft tokens. 
2. By employing a training method that does not affect the standard tokens, the model can acquire parallel decoding abilities atop its original framework with minimal training overhead. 
3. We have designed the ”two-step-draft-then-verify” generation strategies using both greedy search and nucleus sampling. 
Experiments conducted on the CodeLlama-13B and 7B models have yielded speed increases of over **3.5X** and **3.0X**, respectively.
The fine-tuned model and code will be open source.



## Install

```shell
conda create -n sdsat python=3.10
conda activate sdsat

pip install -r requirements.txt
```

## Inference

Execute the following code for speed testing.

1. Greedy Search 
Please Note that at least 40G memory is required for the inference test.
```shell
# greedy sampling of SDSAT-7B on GPU 0,1,2,3
bash start_greedy.sh ainergy/CodeLlama-SDSAT_L5_7B 0,1,2,3

# greedy sampling of SDSAT-13B on GPU 0,1,2,3
bash start_greedy.sh ainergy/CodeLlama-SDSAT_L7_13B 0,1,2,3
```

2. Nucleus Sampling
Please Note that at least 80G memory is required for the inference test.
```shell
# nucleus sampling of SDSAT-7B with temperature 0.2 on GPU 0,1,2,3
bash start_nucleus.sh ainergy/CodeLlama-SDSAT_L5_7B 0,1,2,3 0.2

# nucleus sampling of SDSAT-7B with temperature 1.0 on GPU 0,1,2,3
bash start_nucleus.sh ainergy/CodeLlama-SDSAT_L5_7B 0,1,2,3 1.0
```

3. View test results

The speed test results will be stored in `./results`, go check it

## Model sources

🤯 `https://huggingface.co/ainergy/CodeLlama-SDSAT_L5_7B`

🤯 `https://huggingface.co/ainergy/CodeLlama-SDSAT_L7_13B`


## Citation

```bibtex
@misc{liu2024sdsat,
      title={SDSAT: Accelerating LLM Inference through Speculative Decoding with Semantic Adaptive Tokens}, 
      author={Chengbo Liu and Yong Zhu},
      year={2024},
      eprint={2403.18647},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

