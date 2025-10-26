from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaLlamaConfig
from .language_model.llava_qwen import LlavaQWenForCausalLM, LlavaQWenConfig

import transformers
a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 37:
    from .language_model.llava_qwen2 import LlavaQwen2ForCausalLM, LlavaQwen2Config
    from .language_model.llava_qwen3 import LlavaQwen3ForCausalLM, LlavaQwen3Config
