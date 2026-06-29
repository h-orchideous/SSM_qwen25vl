import json
import os
from collections import OrderedDict

import datasets
import numpy as np
from loguru import logger as eval_logger


def doc_to_visual(doc):
    # Resolve relative video path against user-provided local dataset root.
    base_dir = os.getenv("VSI_SUPER_COUNT_ROOT", "")
    video_path = doc["video_path"]
    if base_dir:
        video_path = os.path.join(base_dir, video_path)
    doc["video_path"] = video_path
    return [doc["video_path"]]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    question = "These are frames of a video.\n" + question + "\nPlease answer the question using a single word or phrase."
    return question


def process_docs_streaming_10mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["split"] == "10mins_streaming")


def process_docs_streaming_30mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["split"] == "30mins_streaming")


def process_docs_streaming_60mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["split"] == "60mins_streaming")


def process_docs_streaming_120mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["split"] == "120mins_streaming")


def process_docs_streaming_240mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["split"] == "240mins_streaming")


def abs_dist_norm(pred, target):
    try:
        return abs(pred - target) / target
    except BaseException:
        return 0.0


def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy


def process_results(doc, results):
    doc["prediction"] = results[0]
    values = json.loads(results[0])

    accs = []
    for streaming_output, answer in zip(values, doc["answers"]):
        accs.append(mean_relative_accuracy(streaming_output, answer, start=0.5, end=0.95, interval=0.05))
    doc["accuracy"] = accs

    return {"score": doc}


def aggregate_results(docs):
    accs = []
    for doc in docs:
        accs.extend(doc["accuracy"])

    accuracy = sum(accs) / len(accs) * 100.0
    accuracy = accuracy.mean().item()

    outputs = OrderedDict()
    outputs["Overall"] = accuracy

    tabulated_keys = ", ".join([k for k in outputs.keys()])
    tabulated_results = ", ".join([f"{v:.3f}" if isinstance(v, float) else str(v) for v in outputs.values()])
    eval_logger.info(f"Tabulated results: {tabulated_keys}")
    eval_logger.info(f"Tabulated results: {tabulated_results}")
    outputs["Tabulated Keys"] = tabulated_keys
    outputs["Tabulated Results"] = tabulated_results

    return outputs
