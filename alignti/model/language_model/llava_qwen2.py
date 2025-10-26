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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM

from .qwen2.modeling_qwen2 import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
# from transformers.modeling_outputs import CausalLMOutputWithPast
from ..utils import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import torch.distributed as dist
from alignti.eagle.model.configs import EConfig
# from alignti.eagle.model.cnets import Model as MTPModel
from alignti.eagle.model.cnets_qwen import Model as MTPModel
from safetensors import safe_open


class LlavaQwen2Config(Qwen2Config):
    model_type = "llava_qwen2"


class LlavaQwen2Model(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwen2Config

    def __init__(self, config: Qwen2Config):
        super(LlavaQwen2Model, self).__init__(config)


class LlavaQwen2ForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = LlavaQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # embedding_layer = self.model.embed_tokens
        # mtp_config = EConfig.from_pretrained('/ossfs/workspace/LLaVA-MoD/alignti/config/mtpconfig/config.json')
        # self.mtp_model = MTPModel(mtp_config, emb_layer=embedding_layer)

        # # 从'/root/llava-checkpoints/distill/llavaqwen-2-0.5b-vis-res-sft-vis-res-mtp-distill'读取mtp_model的权重，其中好包括一些其它参数，筛选一下
        # # 加载根目录下的 model.safetensors
        # checkpoint_path = "/root/llava-checkpoints/distill/llavaqwen-2-0.5b-vis-res-sft-vis-res-mtp-distill/model.safetensors"
        
        # # 加载 safetensors 文件
        # with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        #     mtp_state_dict = {
        #         key.replace("mtp_model.", "", 1): f.get_tensor(key)
        #         for key in f.keys() 
        #         if key.startswith("mtp_model.")
        #     }
        
        # import pdb; pdb.set_trace()
        # # 加载到模型
        # self.mtp_model.load_state_dict(mtp_state_dict, strict=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_model(self):
        return self.model
    
    def get_mtp_model(self):
        return self.mtp_model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            mtp = False,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # import ipdb
        # ipdb.set_trace()
        # print(f'rank {dist.get_rank()}', 'before prepare_inputs_labels_for_multimodal')
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                vision_token_tag
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )
            if inputs_embeds is not None:
                _inputs_embeds = inputs_embeds.clone()
        
        # dist.barrier()
        # print(f'rank {dist.get_rank()}', 'after prepare_inputs_labels_for_multimodal')
        out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mtp=False,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        # dist.barrier()
        # print(f'rank {dist.get_rank()}', 'after LLM')
        # return out, vision_token_tag
        # if mtp:
        # return _inputs_embeds, out, vision_token_tag, images
        return out
        # return out

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs


AutoConfig.register("llava_qwen2", LlavaQwen2Config)
AutoModelForCausalLM.register(LlavaQwen2Config, LlavaQwen2ForCausalLM)
