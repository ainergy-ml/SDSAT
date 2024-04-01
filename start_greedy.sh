#! /bin/bash

MODEL_PATH=$1
GPU_IDS=$2

SATOKEN="32011,32011,32011,32011,32011,32011,32011,32011,32011,32011,32011,32011,32011,32011,32011"
KLIST="0,1,2,3,4,5,6,7,8,9,10,11,12,13"

MODEL_NAME=$(basename "$MODEL_PATH")
SAMPLE_METHOD="greedy"

IFS=',' read -ra ID_ARRAY <<< "$GPU_IDS"
for ID in "${ID_ARRAY[@]}"; do
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
