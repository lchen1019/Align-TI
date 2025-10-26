#!/bin/bash

MODEL_NAME='model_name'
MODEL_PATH='path/to/model'

CONV="qwen"
EVAL="playground/data/eval"

deepspeed --include localhost:4 --master_port 20030 alignti/eval/model_vqa_loader.py \
    --model-path ${MODEL_PATH} \
    --question-file ${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ${EVAL}/textvqa/train_images \
    --answers-file ${EVAL}/textvqa/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

python3 -m alignti.eval.eval_textvqa \
    --annotation-file ${EVAL}/textvqa/TextVQA_0.5.1_val.json \
    --result-file ${EVAL}/textvqa/answers/${MODEL_NAME}.jsonl
