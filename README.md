# SDSAT
The official implementation of "SDSAT: Accelerating LLM Inference through Speculative Decoding with Semantic Adaptive Tokens"

Including the decoding method of both greedy search and nucleus sampling.


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

