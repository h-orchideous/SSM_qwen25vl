#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from accelerate.utils import merge_fsdp_weights


CONFIG_FILES = (
    "config.json",
    "configuration.json",
    "generation_config.json",
    "README.md",
)

PROCESSOR_FILES = (
    "added_tokens.json",
    "chat_template.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def copy_existing(src_dir: Path, dst_dir: Path, names: tuple[str, ...]) -> None:
    for name in names:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge an Accelerate FSDP checkpoint into a loadable HF directory.")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to checkpoint-NNN containing pytorch_model_fsdp_0.")
    parser.add_argument("--output_dir", required=True, help="Output HuggingFace checkpoint directory.")
    parser.add_argument("--base_model_dir", required=True, help="Base Qwen2.5-VL checkpoint directory for config files.")
    parser.add_argument(
        "--processor_dir",
        default=None,
        help="Directory containing tokenizer/processor files. Defaults to the parent of checkpoint_dir.",
    )
    parser.add_argument("--unsafe_pickle", action="store_true", help="Write pytorch_model.bin instead of safetensors.")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    fsdp_dir = checkpoint_dir / "pytorch_model_fsdp_0"
    output_dir = Path(args.output_dir).expanduser().resolve()
    base_model_dir = Path(args.base_model_dir).expanduser().resolve()
    processor_dir = Path(args.processor_dir).expanduser().resolve() if args.processor_dir else checkpoint_dir.parent

    if not fsdp_dir.is_dir():
        raise FileNotFoundError(f"FSDP model directory not found: {fsdp_dir}")
    if not base_model_dir.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")
    if not processor_dir.is_dir():
        raise FileNotFoundError(f"Processor directory not found: {processor_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge_fsdp] merge {fsdp_dir} -> {output_dir}")
    merge_fsdp_weights(
        checkpoint_dir=str(fsdp_dir),
        output_path=str(output_dir),
        safe_serialization=not args.unsafe_pickle,
        remove_checkpoint_dir=False,
    )

    copy_existing(base_model_dir, output_dir, CONFIG_FILES)
    copy_existing(processor_dir, output_dir, PROCESSOR_FILES)
    print(f"[merge_fsdp] wrote loadable checkpoint: {output_dir}")


if __name__ == "__main__":
    main()
