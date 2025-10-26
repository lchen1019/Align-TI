#!/bin/bash

MODEL_NAME='model_name'
MODEL_PATH='path/to/model'

CONV="qwen"
EVAL="playground/data/eval"
IMAGE_FOLDER="playground/data/eval/pope/val2014"

deepspeed --include localhost:2 --master_port 20013 alignti/eval/model_vqa_loader.py \
    --model-path ${MODEL_PATH} \
    --question-file ${EVAL}/pope/llava_pope_test.jsonl \
    --image-folder ${IMAGE_FOLDER} \
    --answers-file ${EVAL}/pope/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

python3 alignti/eval/eval_pope.py \
    --annotation-dir ${EVAL}/pope/coco \
    --question-file ${EVAL}/pope/llava_pope_test.jsonl \
    --result-file ${EVAL}/pope/answers/${MODEL_NAME}.jsonl
