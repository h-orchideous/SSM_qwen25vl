import json
import os
import re
import shutil
from collections import OrderedDict

import datasets
import numpy as np
from loguru import logger as eval_logger

from lmms_eval.models.model_utils.qwen_chunk_utils import (
    build_chunk_specs,
    load_chunk_config,
    materialize_chunk_visual,
)


_chunk_manifest_cache = None
_chunk_manifest_path = None
_stream_chunk_cache = {}
_stream_chunk_dir_by_video = {}
_stream_video_reader_cache = {}


def _truthy_env(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return str(value).lower() in {"1", "true", "yes", "on"}


def _resolve_single_video_target() -> str:
    target = os.getenv("VSC_SINGLE_VIDEO_PATH", "")
    if not target:
        return ""
    if os.path.isabs(target):
        return os.path.normpath(target)
    base_dir = os.getenv("VSI_SUPER_COUNT_ROOT", "")
    if base_dir:
        return os.path.normpath(os.path.join(base_dir, target))
    return os.path.normpath(target)


def _load_chunk_manifest():
    global _chunk_manifest_cache, _chunk_manifest_path
    manifest_path = os.getenv("VSC_PRECHUNK_MANIFEST", "")
    if not manifest_path:
        return None
    if _chunk_manifest_cache is not None and _chunk_manifest_path == manifest_path:
        return _chunk_manifest_cache
    with open(manifest_path, "r", encoding="utf-8") as handle:
        _chunk_manifest_cache = json.load(handle)
    _chunk_manifest_path = manifest_path
    return _chunk_manifest_cache


def _materialize_stream_chunks(video_path: str):
    cached = _stream_chunk_cache.get(video_path)
    if cached is not None:
        return cached

    chunk_cfg = load_chunk_config(max_num_frames_fallback=-1)
    if not chunk_cfg.enabled:
        _stream_chunk_cache[video_path] = [video_path]
        return [video_path]

    output_root = os.getenv("VSC_CHUNK_OUTPUT_ROOT", "")
    if not output_root:
        output_root = os.path.join(os.getcwd(), ".cache", "vsc_chunks")

    safe_video = video_path.lstrip(os.sep).replace("/", "__")
    chunk_dir = os.path.join(output_root, "per_video_stream", safe_video)
    os.makedirs(chunk_dir, exist_ok=True)

    chunk_specs = build_chunk_specs(video_path, chunk_cfg, video_reader_cache=_stream_video_reader_cache)
    chunk_paths = []
    for chunk_spec in chunk_specs:
        chunk_visual = materialize_chunk_visual(
            chunk_spec,
            max_pixels=1,
            min_pixels=1,
            fps=1.0,
            encode_mode=chunk_cfg.encode_mode,
            log_prefix="qwen_vsc_task_stream",
            chunk_dir=chunk_dir,
        )
        if chunk_visual is None:
            continue
        chunk_paths.append(chunk_visual["video"])

    if not chunk_paths:
        chunk_paths = [video_path]
    _stream_chunk_cache[video_path] = chunk_paths
    _stream_chunk_dir_by_video[video_path] = chunk_dir
    eval_logger.info(f"[qwen_vsc_task_stream] prepared video={os.path.basename(video_path)} chunks={len(chunk_paths)}")
    return chunk_paths


def _cleanup_stream_chunks_for_video(video_path: str):
    chunk_dir = _stream_chunk_dir_by_video.pop(video_path, None)
    _stream_chunk_cache.pop(video_path, None)

    if not chunk_dir:
        return
    if _truthy_env("KEEP_CHUNKS", "0"):
        eval_logger.info(f"[qwen_vsc_task_stream] kept video={os.path.basename(video_path)}")
        return

    shutil.rmtree(chunk_dir, ignore_errors=True)
    eval_logger.info(f"[qwen_vsc_task_stream] cleaned video={os.path.basename(video_path)}")


def doc_to_visual(doc):
    base_dir = os.getenv("VSI_SUPER_COUNT_ROOT", "")
    video_path = doc["video_path"]
    if base_dir:
        video_path = os.path.join(base_dir, video_path)
    video_path = os.path.normpath(video_path)
    doc["video_path"] = video_path

    if _truthy_env("VSC_STREAMING_CHUNK_MODE", "0"):
        return _materialize_stream_chunks(video_path)

    manifest = _load_chunk_manifest()
    if manifest and video_path in manifest:
        return manifest[video_path]
    return [doc["video_path"]]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    question = "These are frames of a video.\n" + question + "\nPlease answer the question using a single word or phrase."
    return question


def process_docs_streaming_10mins(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.filter(lambda x: x["split"] == "10mins_streaming")
    return _filter_single_video(dataset)


def process_docs_streaming_30mins(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.filter(lambda x: x["split"] == "30mins_streaming")
    return _filter_single_video(dataset)


def process_docs_streaming_60mins(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.filter(lambda x: x["split"] == "60mins_streaming")
    return _filter_single_video(dataset)


def process_docs_streaming_120mins(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.filter(lambda x: x["split"] == "120mins_streaming")
    return _filter_single_video(dataset)


def process_docs_streaming_240mins(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.filter(lambda x: x["split"] == "240mins_streaming")
    return _filter_single_video(dataset)


def _filter_single_video(dataset: datasets.Dataset) -> datasets.Dataset:
    single_target = _resolve_single_video_target()
    if not single_target:
        return dataset

    base_dir = os.getenv("VSI_SUPER_COUNT_ROOT", "")

    def _keep_doc(doc):
        path = doc["video_path"]
        if base_dir and not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        return os.path.normpath(path) == single_target

    filtered = dataset.filter(_keep_doc)
    eval_logger.info(f"[qwen_vsc_task] single_video_target={single_target} filtered_count={len(filtered)}")
    return filtered


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


def _parse_prediction_values(prediction):
    if prediction is None:
        return []

    if isinstance(prediction, list):
        return prediction

    text = str(prediction).strip()
    if not text:
        return []

    try:
        loaded = json.loads(text)
        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, (int, float)):
            return [loaded]
    except json.JSONDecodeError:
        pass

    values = []
    for match in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


def process_results(doc, results):
    base_dir = os.getenv("VSI_SUPER_COUNT_ROOT", "")
    video_path = doc["video_path"]
    if base_dir and not os.path.isabs(video_path):
        video_path = os.path.join(base_dir, video_path)
    video_path = os.path.normpath(video_path)

    doc["prediction"] = results[0]
    values = _parse_prediction_values(results[0])
    if not values:
        eval_logger.warning(
            f"[qwen_vsc_task] failed to parse prediction as JSON/numeric values. video={doc.get('video_path', '')} prediction={str(results[0])[:120]!r}"
        )

    accs = []
    for streaming_output, answer in zip(values, doc["answers"]):
        accs.append(mean_relative_accuracy(streaming_output, answer, start=0.5, end=0.95, interval=0.05))
    doc["accuracy"] = accs

    if _truthy_env("VSC_STREAMING_CHUNK_MODE", "0"):
        _cleanup_stream_chunks_for_video(video_path)

    return {"score": doc}


def aggregate_results(docs):
    accs = []
    for doc in docs:
        accs.extend(doc["accuracy"])

    if accs:
        accuracy = sum(accs) / len(accs) * 100.0
        accuracy = accuracy.mean().item()
    else:
        accuracy = 0.0

    outputs = OrderedDict()
    outputs["Overall"] = accuracy

    tabulated_keys = ", ".join([k for k in outputs.keys()])
    tabulated_results = ", ".join([f"{v:.3f}" if isinstance(v, float) else str(v) for v in outputs.values()])
    eval_logger.info(f"Tabulated results: {tabulated_keys}")
    eval_logger.info(f"Tabulated results: {tabulated_results}")
    outputs["Tabulated Keys"] = tabulated_keys
    outputs["Tabulated Results"] = tabulated_results

    return outputs
