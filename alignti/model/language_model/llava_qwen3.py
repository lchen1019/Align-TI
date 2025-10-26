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

from .qwen3.modeling_qwen3 import Qwen3Config, Qwen3Model, Qwen3ForCausalLM
# from transformers.modeling_outputs import CausalLMOutputWithPast
from ..utils import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import torch.distributed as dist
from alignti.eagle.model.configs import EConfig
# from alignti.eagle.model.cnets import Model as MTPModel
from alignti.eagle.model.cnets_qwen import Model as MTPModel
from safetensors import safe_open


class LlavaQwen3Config(Qwen3Config):
    model_type = "llava_qwen3"


class LlavaQwen3Model(LlavaMetaModel, Qwen3Model):
    config_class = LlavaQwen3Config

    def __init__(self, config: Qwen3Config):
        super(LlavaQwen3Model, self).__init__(config)


class LlavaQwen3ForCausalLM(Qwen3ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwen3Config

    def __init__(self, config):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = LlavaQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
    
    def get_model(self):
        return self.model

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
        
        out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mtp=mtp,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # return out, vision_token_tag
        # return _inputs_embeds, out, vision_token_tag, images
        return out

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


AutoConfig.register("llava_qwen3", LlavaQwen3Config)
AutoModelForCausalLM.register(LlavaQwen3Config, LlavaQwen3ForCausalLM)
