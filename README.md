# SDSAT
The official implementation of "SDSAT: Accelerating LLM Inference through Speculative Decoding with Semantic Adaptive Tokens"


## Speed test

1. Install

```shell
conda create -n sdsat python=3.10
conda activate sdsat

pip install -r requirements.txt
```

2. Inference


```shell
# greedy sampling
bash start_greedy.sh

# nucleus sampling
bash start_nucleus.sh
```

3. View test results

The speed results are stored in `./results`, go check it

