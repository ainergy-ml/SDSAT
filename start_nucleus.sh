#! /bin/bash

MODEL_PATH="ainergy/CodeLlama-SDSAT_L5_7B"
CUDA_IDS=(0 1 2 3 4 5 6 7)

SATOKEN="32011,32011,32011,32011,32011,32011,32011,32011,32011,32011,32011,32011,32011,32011,32011"
KLIST="0,5,7,9,11,13"
TEMP=0.2

SUFFIX=${TEMP/./_}
MODEL_NAME=$(basename "$MODEL_PATH")
SAMPLE_METHOD="nucleus"

for ID in "${CUDA_IDS[@]}"; do
    export CUDA_VISIBLE_DEVICES="$ID"
    RESULT_DIR=`pwd`/results/${MODEL_NAME}/${SAMPLE_METHOD}_${SUFFIX}/${SAMPLE_METHOD}_GPU_$ID

    (
        python3 main.py \
            --device=cuda \
            --data_name=humaneval \
            --result_dir=$RESULT_DIR \
            --model_name=codellama \
            --model_path=$MODEL_PATH \
            --sa_tokens=$SATOKEN \
            --do_sample \
            --temperature=$TEMP \
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
            --do_sample \
            --temperature=$TEMP \
            --max_new_tokens=50 \
            --k_list=$KLIST \
            --data_limit=0.03
    ) &
done
wait

python3 plot.py \
    --data_path=`pwd`/results/${MODEL_NAME}/${SAMPLE_METHOD}_${SUFFIX} \
    --output_path=`pwd`/results/${MODEL_NAME}/${SAMPLE_METHOD}_${SUFFIX}
