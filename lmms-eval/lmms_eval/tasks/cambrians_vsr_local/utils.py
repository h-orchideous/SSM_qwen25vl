import os
import json
import shutil
from collections import OrderedDict

import datasets
from loguru import logger as eval_logger

from lmms_eval.models.model_utils.qwen_chunk_utils import (
    load_chunk_config,
    build_chunk_specs,
    materialize_chunk_visual,
)


_chunk_manifest_cache = None
_chunk_manifest_path = None
_stream_chunk_cache = {}
_stream_chunk_dir_by_video = {}
_stream_video_reader_cache = {}


def _resolve_single_video_target() -> str:
    target = os.getenv("VSR_SINGLE_VIDEO_PATH", "")
    if not target:
        return ""
    if os.path.isabs(target):
        return os.path.normpath(target)
    base_dir = os.getenv("VSI_SUPER_RECALL_ROOT", "")
    if base_dir:
        return os.path.normpath(os.path.join(base_dir, target))
    return os.path.normpath(target)


def _load_chunk_manifest():
    global _chunk_manifest_cache, _chunk_manifest_path
    manifest_path = os.getenv("VSR_PRECHUNK_MANIFEST", "")
    if not manifest_path:
        return None
    if _chunk_manifest_cache is not None and _chunk_manifest_path == manifest_path:
        return _chunk_manifest_cache
    with open(manifest_path, "r", encoding="utf-8") as handle:
        _chunk_manifest_cache = json.load(handle)
    _chunk_manifest_path = manifest_path
    return _chunk_manifest_cache


def _truthy_env(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return str(value).lower() in {"1", "true", "yes", "on"}


def _materialize_stream_chunks(video_path: str):
    cached = _stream_chunk_cache.get(video_path)
    if cached is not None:
        return cached

    chunk_cfg = load_chunk_config(max_num_frames_fallback=-1)
    if not chunk_cfg.enabled:
        _stream_chunk_cache[video_path] = [video_path]
        return [video_path]

    output_root = os.getenv("VSR_CHUNK_OUTPUT_ROOT", "")
    if not output_root:
        output_root = os.path.join(os.getcwd(), ".cache", "vsr_chunks")

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
            log_prefix="vsr_task_stream",
            chunk_dir=chunk_dir,
        )
        if chunk_visual is None:
            continue
        chunk_paths.append(chunk_visual["video"])

    if not chunk_paths:
        chunk_paths = [video_path]
    _stream_chunk_cache[video_path] = chunk_paths
    _stream_chunk_dir_by_video[video_path] = chunk_dir
    eval_logger.info(f"[vsr_task_stream] prepared video={os.path.basename(video_path)} chunks={len(chunk_paths)}")
    return chunk_paths


def _cleanup_stream_chunks_for_video(video_path: str):
    chunk_dir = _stream_chunk_dir_by_video.pop(video_path, None)
    _stream_chunk_cache.pop(video_path, None)

    if not chunk_dir:
        return
    if _truthy_env("KEEP_CHUNKS", "0"):
        eval_logger.info(f"[vsr_task_stream] kept video={os.path.basename(video_path)}")
        return

    shutil.rmtree(chunk_dir, ignore_errors=True)
    eval_logger.info(f"[vsr_task_stream] cleaned video={os.path.basename(video_path)}")


def doc_to_visual(doc):
    # Resolve relative video path against user-provided dataset root.
    base_dir = os.getenv("VSI_SUPER_RECALL_ROOT", "")
    video_path = doc["video_path"]
    if base_dir:
        video_path = os.path.join(base_dir, video_path)
    video_path = os.path.normpath(video_path)
    doc["video_path"] = video_path

    if _truthy_env("VSR_STREAMING_CHUNK_MODE", "0"):
        return _materialize_stream_chunks(video_path)

    manifest = _load_chunk_manifest()
    if manifest and video_path in manifest:
        return manifest[video_path]
    return [doc["video_path"]]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    options = doc["options"]
    question = question + "\nOptions:\n" + "\n".join(options) + "\nAnswer with the option's letter from the given choices directly."
    return question


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    single_target = _resolve_single_video_target()
    if not single_target:
        return dataset

    base_dir = os.getenv("VSI_SUPER_RECALL_ROOT", "")

    def _keep_doc(doc):
        path = doc["video_path"]
        if base_dir and not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        return os.path.normpath(path) == single_target

    filtered = dataset.filter(_keep_doc)
    eval_logger.info(f"[vsr_task] single_video_target={single_target} filtered_count={len(filtered)}")
    return filtered


def fuzzy_matching(pred):
    return pred.split(" ")[0].rstrip(".").strip()


def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0


def process_results(doc, results):
    base_dir = os.getenv("VSI_SUPER_RECALL_ROOT", "")
    video_path = doc["video_path"]
    if base_dir and not os.path.isabs(video_path):
        video_path = os.path.join(base_dir, video_path)
    video_path = os.path.normpath(video_path)

    doc["prediction"] = results[0]
    doc["accuracy"] = exact_match(fuzzy_matching(doc["prediction"]), doc["answer"])

    if _truthy_env("VSR_STREAMING_CHUNK_MODE", "0"):
        _cleanup_stream_chunks_for_video(video_path)

    return {"score": doc}


def aggregate_results(docs):
    total, correct = 0, 0
    for doc in docs:
        total += 1
        correct += doc["accuracy"]

    accuracy = correct / total * 100.0 if total > 0 else 0.0
    eval_logger.info(f"Overall accuracy: {accuracy:.3f}")

    outputs = OrderedDict()
    outputs["Overall"] = accuracy

    tabulated_keys = ", ".join([k for k in outputs.keys()])
    tabulated_results = ", ".join([f"{v:.3f}" if isinstance(v, float) else str(v) for v in outputs.values()])
    eval_logger.info(f"Tabulated results: {tabulated_keys}")
    eval_logger.info(f"Tabulated results: {tabulated_results}")
    outputs["Tabulated Keys"] = tabulated_keys
    outputs["Tabulated Results"] = tabulated_results

    return outputs
