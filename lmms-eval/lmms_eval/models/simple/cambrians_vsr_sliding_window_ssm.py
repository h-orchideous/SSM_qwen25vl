from lmms_eval.api.registry import register_model

from lmms_eval.models.simple.cambrians_vsr_sliding_window import CambrianS_VSR


@register_model("cambrians_vsr_sliding_window_ssm")
class CambrianS_VSR_SlidingWindow_SSM(CambrianS_VSR):
    pass
