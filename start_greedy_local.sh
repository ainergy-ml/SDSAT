#! /bin/bash

MODEL_PATH="/store4/code_models/codellama_pretrain_lcb/multi-204/SDSAT_L7_13B"
# CUDA_IDS=(0 1 2 3 4 5 6 7)
CUDA_IDS=(1)

SATOKEN="32011,32011,32011,32011,32011,32011,32011,32011,32011,32011,32011,32011,32011,32011,32011"
KLIST="0,1,2,3,4,5,6,7,8,9,10,11,12,13"

MODEL_NAME=$(basename "$MODEL_PATH")
SAMPLE_METHOD="greedy"

for ID in "${CUDA_IDS[@]}"; do
    export CUDA_VISIBLE_DEVICES="$ID"
    RESULT_DIR=`pwd`/results/${MODEL_NAME}/${SAMPLE_METHOD}/${SAMPLE_METHOD}_GPU_$ID

    (
        python3 main.py \
            --device=cuda \
            --data_name=humaneval \
            --result_dir=$RESULT_DIR \
            --model_name=codellama \
            --model_path=$MODEL_PATH \
            --sa_tokens=$SATOKEN \
            --use_cache \
            --max_new_tokens=512 \
            --k_list=$KLIST \
            --data_limit=0.5

        python3 main.py \
            --device=cuda \
            --data_name=multiple-infilling \
            --result_dir=$RESULT_DIR \
            --model_name=codellama \
            --model_path=$MODEL_PATH \
            --sa_tokens=$SATOKEN \
            --use_cache \
            --max_new_tokens=50 \
            --k_list=$KLIST \
            --data_limit=0.03
    ) &
done
wait

python3 plot.py \
    --data_path=`pwd`/results/${MODEL_NAME}/${SAMPLE_METHOD} \
    --output_path=`pwd`/results/${MODEL_NAME}/${SAMPLE_METHOD}
