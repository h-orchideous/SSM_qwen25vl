import json
import re
from typing import List

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import parse_reasoning_model_answer
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL


@register_model("qwen2_5_vl_vsc")
class Qwen2_5_VL_VSC(Qwen2_5_VL):
    """Qwen2.5-VL VSC baseline.

    The base Qwen2.5-VL path produces one final answer. VSC streaming metrics
    expect a JSON list aligned with the dataset's answers, so this wrapper
    repeats the final count for each required output slot.
    """

    def _format_count_answer_for_streaming_metric(self, answer: str, num_outputs: int) -> str:
        value = self._extract_count(answer)
        return json.dumps([value for _ in range(max(1, int(num_outputs)))])

    def _extract_count(self, answer: str) -> float:
        clean = parse_reasoning_model_answer(answer).strip()
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", clean)
        if not match:
            return 0.0
        try:
            return float(match.group(0))
        except ValueError:
            return 0.0

    def _aggregate_chunk_answers(self, answers: List[str]) -> str:
        if not answers:
            return ""
        values = [self._extract_count(answer) for answer in answers]
        return str(sum(values))

    def generate_until(self, requests: List[Instance]) -> List[str]:
        outputs = super().generate_until(requests)
        formatted = []
        for output, request in zip(outputs, requests):
            _, _, _, doc_id, task, split = request.args
            sample = self.task_dict[task][split][doc_id]
            if hasattr(sample, "get") and sample.get("answers", None) is not None:
                formatted.append(self._format_count_answer_for_streaming_metric(output, len(sample["answers"])))
            else:
                formatted.append(output)
        return formatted
