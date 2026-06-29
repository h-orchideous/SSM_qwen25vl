#!/usr/bin/env python3
import os
import shlex
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def to_bool_str(v: str) -> str:
    return "True" if str(v).lower() in {"1", "true", "yes", "y", "on"} else "False"


def find_available_port(start_port: int, max_tries: int = 100) -> int:
    for offset in range(max_tries):
        port = start_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range [{start_port}, {start_port + max_tries - 1}]")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    conda_home = Path(env("CONDA_HOME", str(Path.home() / "miniconda3")))
    conda_env_name = env("CAMBRIAN_CONDA_ENV", "cambrian-gpu")
    conda_python = conda_home / "envs" / conda_env_name / "bin" / "python"
    if not conda_python.is_file():
        print(f"[ERROR] Expected conda env python not found: {conda_python}")
        return 1

    data_path = env("DATA_PATH", "/data1/ZhangHuayu/datasets/VSI-Train-10k/vsi_train_10k.jsonl")
    image_folder = env("IMAGE_FOLDER", "/data1/ZhangHuayu/datasets/VSI-Train-10k")
    model_name_or_path = env("MODEL_NAME_OR_PATH", "/data1/ZhangHuayu/models/Qwen2.5-7B-Instruct")

    if not Path(data_path).is_file():
        print(f"[ERROR] DATA_PATH file not found: {data_path}")
        return 1
    if not Path(image_folder).is_dir():
        print(f"[ERROR] IMAGE_FOLDER directory not found: {image_folder}")
        return 1

    output_dir = env("OUTPUT_DIR", str(repo_root / "outputs" / "cambrians_7b_s1"))
    logs_dir = Path(env("LOGS_DIR", str(repo_root / "outputs" / "train_logs")))
    log_file = env("LOG_FILE", str(logs_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    run_name = env("RUN_NAME", "cambrians_7b_s1_local_oom_safe")
    report_to = env("REPORT_TO", "none")
    launcher = env("LAUNCHER", "fsdp")
    strict_launch = to_bool_str(env("STRICT_LAUNCH", "True"))

    gpu_ids = env("GPU_IDS", "0,1,2,3")
    visible_gpus = [g.strip() for g in gpu_ids.split(",") if g.strip()]
    world_size = len(visible_gpus) if visible_gpus else 1
    use_fsdp = env("USE_FSDP", "1" if world_size > 1 else "0")
    requested_master_port = int(env("MASTER_PORT", "29503"))
    master_port = find_available_port(requested_master_port)
    if master_port != requested_master_port:
        print(f"[WARN] MASTER_PORT {requested_master_port} is in use, fallback to {master_port}")
    fsdp_config = env("FSDP_CONFIG", "fsdp_config_cuda.json")
    per_device_train_batch_size = env("PER_DEVICE_TRAIN_BATCH_SIZE", "1")
    gradient_accumulation_steps = env("GRADIENT_ACCUMULATION_STEPS", "8")
    dataloader_num_workers = env("DATALOADER_NUM_WORKERS", "4")

    bf16 = to_bool_str(env("BF16", "True"))
    tf32 = to_bool_str(env("TF32", "True"))
    bits = env("BITS", "4")
    double_quant = to_bool_str(env("DOUBLE_QUANT", "True"))
    quant_type = env("QUANT_TYPE", "nf4")

    is_quantized = bits in {"4", "8"}

    if strict_launch == "True":
        if launcher != "fsdp":
            print(f"[ERROR] STRICT_LAUNCH requires LAUNCHER=fsdp, got: {launcher}")
            return 1
        if gpu_ids != "0,1,2,3" or world_size != 4:
            print(f"[ERROR] STRICT_LAUNCH requires GPU_IDS=0,1,2,3 (world_size=4), got: {gpu_ids} (world_size={world_size})")
            return 1
        if bits != "4":
            print(f"[ERROR] STRICT_LAUNCH requires BITS=4, got: {bits}")
            return 1

    if is_quantized and use_fsdp == "1":
        print("[WARN] Quantized (4/8-bit) run detected, disabling FSDP to avoid FSDP init OOM.")
        use_fsdp = "0"

    video_folder = env("VIDEO_FOLDER", image_folder)
    video_fps = env("VIDEO_FPS", "1")
    video_max_frames = env("VIDEO_MAX_FRAMES", "1")
    video_force_sample = to_bool_str(env("VIDEO_FORCE_SAMPLE", "True"))

    max_images_per_sample = env("MAX_IMAGES_PER_SAMPLE", "16")
    anyres_max_subimages = env("ANYRES_MAX_SUBIMAGES", "4")

    ssm_d_state = env("SSM_D_STATE", "64")
    ssm_fusion_num_heads = env("SSM_FUSION_NUM_HEADS", "8")
    ssm_fusion_bottleneck = env("SSM_FUSION_BOTTLENECK", "23328")
    ssm_layer_sharing = env("SSM_LAYER_SHARING", "group4")
    ssm_use_fast_path = env("SSM_USE_FAST_PATH", "1")

    desired_ssm_frames = int(env("DESIRED_SSM_FRAMES", "64"))
    si_token_len = int(env("SI_TOKEN_LEN", "729"))
    ssm_max_memory_len = env("SSM_MAX_MEMORY_LEN", str(desired_ssm_frames * si_token_len))

    max_seq_len = env("MAX_SEQ_LEN", "4096")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Reduce fragmentation risk for long-running multi-GPU jobs.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    os.environ["CAMBRIAN_LAUNCHER"] = ""
    os.environ["ACCELERATE_BYPASS_DEVICE_MAP"] = "true"
    os.environ["CAMBRIAN_EXPECT_ENV"] = conda_env_name
    os.environ["CAMBRIAN_EXPECT_PYTHON"] = str(conda_python)
    os.environ["CAMBRIAN_EXPECT_MODEL_IMPL"] = "cambrian.model.language_model.qwen2_5_ssm"
    if is_quantized and world_size > 1:
        # Use all visible GPUs via one-process model parallel for quantized runs.
        os.environ["CAMBRIAN_DEVICE_MAP_AUTO"] = "1"
    os.environ.pop("PJRT_DEVICE", None)
    os.environ.pop("GPU_NUM_DEVICES", None)

    train_script = repo_root / "cambrian" / "train" / f"train_{launcher}.py"
    if not train_script.is_file():
        print(f"[ERROR] Launcher script not found: {train_script}")
        return 1

    train_args = [
        "--model_name_or_path", model_name_or_path,
        "--version", "qwen_2",
        "--data_path", data_path,
        "--image_folder", image_folder,
        "--vision_tower_aux_list", '["google/siglip2-so400m-patch14-384"]',
        "--vision_tower_aux_token_len_list", "[729]",
        "--image_position", "14",
        "--vision_hidden_size", "1152",
        "--video_fps", video_fps,
        "--video_max_frames", video_max_frames,
        "--video_force_sample", video_force_sample,
        "--video_folder", video_folder,
        "--connector_only", "True",
        "--unfreeze_mm_vision_tower", "False",
        "--mm_projector_type", "mlp2x_gelu",
        "--mm_projector_lr", "1e-3",
        "--tune_mm_mlp_adapter", "True",
        "--mm_vision_select_layer", "-2",
        "--mm_use_im_start_end", "False",
        "--mm_use_im_patch_token", "False",
        "--mm_use_im_newline_token", "True",
        "--image_aspect_ratio", "pad",
        "--group_by_modality_length", "True",
        "--bf16", bf16,
        "--bits", bits,
        "--double_quant", double_quant,
        "--quant_type", quant_type,
        "--output_dir", output_dir,
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", per_device_train_batch_size,
        "--per_device_eval_batch_size", "4",
        "--gradient_accumulation_steps", gradient_accumulation_steps,
        "--evaluation_strategy", "no",
        "--save_strategy", "steps",
        "--save_steps", "250",
        "--save_total_limit", "1",
        "--learning_rate", "1e-5",
        "--weight_decay", "0.",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--tf32", tf32,
        "--model_max_length", max_seq_len,
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", dataloader_num_workers,
        "--lazy_preprocess", "True",
        "--report_to", report_to,
        "--run_name", run_name,
        "--max_images_per_sample", max_images_per_sample,
        "--anyres_max_subimages", anyres_max_subimages,
        "--si_token_len", str(si_token_len),
        "--miv_token_len", "64",
        "--ssm_d_state", ssm_d_state,
        "--ssm_max_memory_len", ssm_max_memory_len,
        "--ssm_fusion_num_heads", ssm_fusion_num_heads,
        "--ssm_fusion_bottleneck", ssm_fusion_bottleneck,
        "--ssm_layer_sharing", ssm_layer_sharing,
        "--ssm_use_fast_path", ssm_use_fast_path,
    ]

    if use_fsdp == "1":
        fsdp_config_path = repo_root / fsdp_config
        if not fsdp_config_path.is_file():
            print(f"[ERROR] FSDP config not found: {fsdp_config_path}")
            return 1
        train_args.extend(["--fsdp", "full_shard", "--fsdp_config", str(fsdp_config_path)])

    use_torchrun = world_size > 1 and not is_quantized

    if not use_torchrun:
        # Ensure Accelerate does not pick up stale distributed variables from a previous torchrun shell.
        for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT"]:
            os.environ.pop(key, None)

    if use_torchrun:
        args = [
            str(conda_python),
            "-m",
            "torch.distributed.run",
            "--nproc_per_node", str(world_size),
            "--master_port", str(master_port),
            str(train_script),
            *train_args,
        ]
    else:
        args = [str(conda_python), str(train_script), *train_args]

    print("[INFO] OOM-safe launcher enabled")
    print(f"[INFO] Using conda env: {conda_env_name}")
    print(f"[INFO] Using python: {conda_python}")
    print(f"[INFO] CUDA_VISIBLE_DEVICES={gpu_ids}")
    print(f"[INFO] World size: {world_size}")
    print(f"[INFO] Launch mode: {'torchrun' if use_torchrun else 'single-process'}")
    print(f"[INFO] USE_FSDP={use_fsdp}")
    print(f"[INFO] FSDP_CONFIG={fsdp_config}")
    print(f"[INFO] STRICT_LAUNCH={strict_launch}")
    print(f"[INFO] PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    print(f"[INFO] Quantization: bits={bits}, double_quant={double_quant}, quant_type={quant_type}")
    print(f"[INFO] Log file: {log_file}")
    print("[INFO] Command:")
    print(" ".join(shlex.quote(x) for x in args))

    with open(log_file, "a", encoding="utf-8") as log_fp:
        log_fp.write("[INFO] OOM-safe launcher command\n")
        log_fp.write(" ".join(shlex.quote(x) for x in args) + "\n\n")

        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_fp.write(line)

        return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
