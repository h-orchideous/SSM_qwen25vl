__all__ = ["CambrianQwenForCausalLM", "CambrianQwenConfig", "Qwen2_5SSMForCausalLM", "Qwen2_5SSMConfig"]


def __getattr__(name):
	if name in __all__:
		from .model.language_model.qwen2_5_ssm import (
			CambrianQwenConfig,
			CambrianQwenForCausalLM,
			Qwen2_5SSMConfig,
			Qwen2_5SSMForCausalLM,
		)

		exports = {
			"CambrianQwenForCausalLM": CambrianQwenForCausalLM,
			"CambrianQwenConfig": CambrianQwenConfig,
			"Qwen2_5SSMForCausalLM": Qwen2_5SSMForCausalLM,
			"Qwen2_5SSMConfig": Qwen2_5SSMConfig,
		}
		return exports[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
