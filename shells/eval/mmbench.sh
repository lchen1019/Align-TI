#!/bin/bash

MODEL_NAME='model_name'
MODEL_PATH='path/to/model'

CONV="qwen"
EVAL="playground/data/eval"
SPLIT="mmbench_dev_20230712"

deepspeed --include localhost:4 --master_port 20029 alignti/eval/model_vqa_mmbench.py \
     --model-path ${MODEL_PATH} \
     --question-file ${EVAL}/mmbench/$SPLIT.tsv \
     --answers-file ${EVAL}/mmbench/answers/$SPLIT/${MODEL_NAME}.jsonl \
     --single-pred-prompt \
     --temperature 0 \
     --conv-mode ${CONV}

mkdir -p ${EVAL}/mmbench/answers_upload/$SPLIT

python3 scripts/convert_mmbench_for_submission.py \
    --annotation-file ${EVAL}/mmbench/$SPLIT.tsv \
    --result-dir ${EVAL}/mmbench/answers/$SPLIT \
    --upload-dir ${EVAL}/mmbench/answers_upload/$SPLIT \
    --experiment ${MODEL_NAME}
