from transformers import AutoConfig, AutoModelForCausalLM

from .cambrian_qwen2_ssm import (
    CambrianQwenConfig as _BaseConfig,
    CambrianQwenForCausalLM as _BaseForCausalLM,
    CambrianQwenModel as _BaseModel,
)


class Qwen2_5SSMConfig(_BaseConfig):
    model_type = "qwen2_5_ssm"


class Qwen2_5SSMModel(_BaseModel):
    config_class = Qwen2_5SSMConfig


class Qwen2_5SSMForCausalLM(_BaseForCausalLM):
    config_class = Qwen2_5SSMConfig


# Backward-compatible aliases.
CambrianQwenConfig = Qwen2_5SSMConfig
CambrianQwenModel = Qwen2_5SSMModel
CambrianQwenForCausalLM = Qwen2_5SSMForCausalLM


AutoConfig.register("qwen2_5_ssm", Qwen2_5SSMConfig)
AutoModelForCausalLM.register(Qwen2_5SSMConfig, Qwen2_5SSMForCausalLM)