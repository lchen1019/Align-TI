#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import torch
import warnings
import shutil
import transformers
a, b, c = transformers.__version__.split('.')[:3]

from alignti.model.language_model.llava_llama import LlavaLlamaForCausalLM
from alignti.model.language_model.llava_qwen import LlavaQWenForCausalLM
if a == '4' and int(b) >= 37:
    from alignti.model.language_model.llava_qwen2 import LlavaQwen2ForCausalLM
    from alignti.model.language_model.llava_qwen3 import LlavaQwen3ForCausalLM

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, GenerationConfig
from alignti.model import *
from alignti.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VID_END_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VIDEO_PATCH_TOKEN
from alignti.model.language_model.qwen.tokenization_qwen import QWenTokenizer



def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto",
                          device="cuda", padding_side="right", merge=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.bfloat16
    
    # import pdb; pdb.set_trace()

    if 'llava' in model_name.lower():
        if 'qwen' in model_name.lower():
            if 'qwen2' in model_name.lower() or 'qwen-2' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                # print(tokenizer)
                model = LlavaQwen2ForCausalLM.from_pretrained(model_path, attn_implementation='sdpa', low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).cuda()
                # import pdb; pdb.set_trace()
                model.config.eos_token_id = tokenizer.eos_token_id

                # model = LlavaQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

                model.generation_config = GenerationConfig.from_pretrained(model_path,
                                                            pad_token_id=tokenizer.pad_token_id)
                model.generation_config.repetition_penalty = None
                model.generation_config.do_sample = False  # use greedy decoding
                model.generation_config.repetition_penalty = 1.0  # disable repetition penalty
            elif 'qwen3' in model_name.lower() or 'qwen-3' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                # import pdb; pdb.set_trace()
                model = LlavaQwen3ForCausalLM.from_pretrained(model_path, attn_implementation='sdpa', low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).cuda()
                # model = LlavaQwen3ForCausalLM.from_pretrained(model_path, attn_implementation='sdpa', low_cpu_mem_usage=True, **kwargs)
                model.config.eos_token_id = tokenizer.eos_token_id
            else:
                tokenizer = QWenTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                model = LlavaQWenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                # print(model)
                model.generation_config = GenerationConfig.from_pretrained(model_path,
                                                                            pad_token_id=tokenizer.pad_token_id)
                # model.generation_config.repetition_penalty = None

                model.generation_config.do_sample = False  # use greedy decoding
                model.generation_config.repetition_penalty = 1.0  # disable repetition penalty
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
            model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding_side=padding_side)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True,
                                                             **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    print("######### model name")
    for name, param in model.named_parameters():
        print(name, param.dtype)

    # ==========================================================================================================
    processor = {'image': None, 'video': None}

    # import ipdb; ipdb.set_trace()
    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        if model.config.mm_image_tower is not None:
            image_tower = model.get_image_tower()
            if not image_tower.is_loaded:
                image_tower.load_model()
            image_tower.to(device=device, dtype=torch.bfloat16)
            image_processor = image_tower.image_processor
            processor['image'] = image_processor

        if model.config.mm_video_tower is not None:
            video_tower = model.get_video_tower()
            if not video_tower.is_loaded:
                video_tower.load_model()
            video_tower.to(device=device, dtype=torch.bfloat16)
            video_processor = video_tower.video_processor
            processor['video'] = video_processor

    # ==========================================================================================================
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, processor, context_len
