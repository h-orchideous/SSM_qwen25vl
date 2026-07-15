# 使用 Qwen2.5-VL 评测 VSR

本文档说明如何使用 `scripts/vsr_qwen25vl.sh` 在本地 VSR 任务上评测 Qwen2.5-VL，以及如何切换不同长视频处理机制。

当前脚本支持三种评测模式：

| `MODEL_VARIANT` | lmms-eval 模型名                | 机制                                          | 默认 checkpoint                                     |
| ----------------- | ------------------------------- | --------------------------------------------- | --------------------------------------------------- |
| `vl`            | `qwen2_5_vl`                  | 原生 Qwen2.5-VL 视频推理，配合 chunk/fps 控制 | `/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct` |
| `sw`            | `qwen_vsr_sliding_window`     | SimpleStream：流式读帧，只保留最近 N 帧 raw frames，最后 query 一次 | `/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct` |
| `ssm`           | `qwen_vsr_sliding_window_ssm` | KV-CSMS 流式机制：每帧先经过 Qwen2.5-VL prefill 产生 decoder KV，窗口外 visual KV 按帧内 H×W 重排后写入 4-direction SSM hidden state，最后用当前 KV 窗口 + SSM fusion query | `/data1/ZhangHuayu/models/Qwen2.5-VL-7B-Instruct` |

## 入口脚本

主评测脚本：

```bash
cd /home/ZhangHuayu/Workspace/cambrian-s/lmms-eval
bash scripts/vsr_qwen25vl.sh
```

共享切片配置：

```bash
scripts/vsr_chunk_settings.sh
```

`vsr_qwen25vl.sh` 会自动 source `vsr_chunk_settings.sh`。评测前可以直接修改该文件，也可以在启动命令前用环境变量覆盖。

默认评测任务是：

```bash
vsr_local_10mins
```

如果要评测其他 VSR split，修改 `scripts/vsr_qwen25vl.sh` 末尾：

```bash
run_eval "vsr_local_10mins"
```

可用的本地任务 YAML 位于 `lmms_eval/tasks/cambrians_vsr_local/`，包括：

```text
vsr_local_10mins
vsr_local_30mins
vsr_local_60mins
vsr_local_120mins
vsr_local_240mins
```

## 环境要求

脚本假定当前环境满足：

- `lmms_eval` 可以被当前 Python 环境 import。
- 已安装 `accelerate`、`transformers`、`torch`、`decord`、`qwen-vl-utils` 以及视频读取相关依赖。
- `nvidia-smi` 可用，用于记录 GPU 利用率、显存和功耗。
- VSR 数据集位于 `/data1/ZhangHuayu/datasets/VSI-SUPER-Recall`。
- 模型 checkpoint 位于默认路径，或通过 `CHECKPOINT` 覆盖。

脚本会设置这些关键环境变量：

```bash
export VSI_SUPER_RECALL_ROOT="/data1/ZhangHuayu/datasets/VSI-SUPER-Recall"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=20480
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

如果没有设置 `CUDA_VISIBLE_DEVICES`，脚本默认使用：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7
```

## 基本启动命令

原生 Qwen2.5-VL：

```bash
cd /home/ZhangHuayu/Workspace/cambrian-s/lmms-eval
MODEL_VARIANT=vl bash scripts/vsr_qwen25vl.sh
```

Qwen2.5-VL 滑动窗口：

```bash
cd /home/ZhangHuayu/Workspace/cambrian-s/lmms-eval
MODEL_VARIANT=sw bash scripts/vsr_qwen25vl.sh
```

Qwen2.5-VL SSM 滑动窗口：

```bash
cd /home/ZhangHuayu/Workspace/cambrian-s/lmms-eval
MODEL_VARIANT=ssm bash scripts/vsr_qwen25vl.sh
```

指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL_VARIANT=sw bash scripts/vsr_qwen25vl.sh
```

指定 checkpoint：

```bash
MODEL_VARIANT=vl \
CHECKPOINT=/path/to/Qwen2.5-VL-7B-Instruct \
bash scripts/vsr_qwen25vl.sh
```

`ssm` 模式现在使用原生 Qwen2.5-VL checkpoint，不再要求 Cambrian 多模态 checkpoint。

## 视频切片配置

长视频通过时间窗口切片进行评测。默认配置在 `scripts/vsr_chunk_settings.sh`：

```bash
export CHUNK_SECONDS=200
export CHUNK_OVERLAP_SECONDS=0
export CHUNK_MAX_NUM_FRAMES=200
export CHUNK_ENCODE_MODE=reencode
export CHUNK_PER_VIDEO_FLOW=1
export CHUNK_TEMP_MODE=1
export KEEP_CHUNKS=0
```

参数含义：

| 参数                      | 说明                                                         |
| ------------------------- | ------------------------------------------------------------ |
| `CHUNK_SECONDS`         | 每个视频切片的时间长度。设置为`0` 表示关闭切片。           |
| `CHUNK_OVERLAP_SECONDS` | 相邻切片之间的时间重叠。                                     |
| `CHUNK_MAX_NUM_FRAMES`  | 每个切片最多采样多少帧。                                     |
| `CHUNK_ENCODE_MODE`     | 切片生成方式，可选`reencode`、`copy`、`auto`。         |
| `CHUNK_PER_VIDEO_FLOW`  | 设置为`1` 时，每个视频边切片边推理，并在该视频结束后清理。 |
| `CHUNK_TEMP_MODE`       | 设置为`1` 时，切片写入本次运行的临时目录。                 |
| `KEEP_CHUNKS`           | 设置为`1` 时保留临时切片，便于调试。                       |

示例：使用更短切片并增加重叠：

```bash
MODEL_VARIANT=vl \
CHUNK_SECONDS=24 \
CHUNK_OVERLAP_SECONDS=12 \
CHUNK_MAX_NUM_FRAMES=32 \
bash scripts/vsr_qwen25vl.sh
```

示例：保留生成的视频切片：

```bash
KEEP_CHUNKS=1 MODEL_VARIANT=sw bash scripts/vsr_qwen25vl.sh
```

## 三种机制的关键参数

### `MODEL_VARIANT=vl`

该模式使用 `qwen2_5_vl`，把切片后的视频窗口送入 Qwen2.5-VL，再由模型路径聚合多个 chunk 的生成结果。

常用覆盖参数：

```bash
MAX_PIXELS=175616
MIN_PIXELS=175616
MAX_NUM_FRAMES=768
VIDEO_FPS=1
USE_CUSTOM_VIDEO_LOADER=True
MAX_IMAGE_SIZE=448
ATTN_IMPLEMENTATION=sdpa
USE_FAST_PROCESSOR=True
```

注意：如果 `VIDEO_FPS=1`、`MAX_NUM_FRAMES<=0` 且关闭切片，脚本会直接报错退出。原因是原生 Qwen2.5-VL 对整段长视频做一次 forward 时，视觉 attention 显存会随序列长度快速增长。

### `MODEL_VARIANT=sw`

该模式使用 `qwen_vsr_sliding_window`，对应 SimpleStream recent-window baseline：视频按 chunk 流式读取，内部只维护最近 `SENSORY_WINDOW_SIZE` 帧 raw frames；窗口满后新帧进入、最旧帧直接丢弃。所有 chunk 读完后，只把最后窗口里的 N 帧作为普通 Qwen2.5-VL video 输入并 query 一次。

常用覆盖参数：

```bash
NUM_FRAMES=-1
MIV_TOKEN_LEN=64
SI_TOKEN_LEN=729
SENSORY_WINDOW_SIZE=140
SLIDING_WINDOW_STRIDE=1
SENSORY_WINDOW_MAX_TOKENS=0
STREAM_VISUAL_MICRO_BATCH_SIZE=1
STREAM_QUERY_MODE=chunk
```

如果未设置 `NUM_FRAMES`，`sw` 模式默认使用 `-1`，表示模型路径内部不再限制每个 chunk 的帧数。此时建议主要通过 `CHUNK_MAX_NUM_FRAMES` 控制切片阶段的帧数。当前 `sw` 不写 KV cache，也不保留窗口外历史；`STREAM_QUERY_MODE` 仅保留为兼容参数，不决定 query 次数。

### `MODEL_VARIANT=ssm`

该模式使用 `qwen_vsr_sliding_window_ssm`。它实现的是 KV-CSMS：视频按 chunk 流式读取，每个采样帧都会先经过 Qwen2.5-VL prefill，产生该帧对应的 decoder `past_key_values`；当前 cache 只保留最近 `SENSORY_WINDOW_SIZE` 帧对应的 KV。窗口外 visual KV 在被裁掉前按该帧视觉 token 的 H×W 网格重排，并沿 4 个方向写入 SSM hidden state。所有 chunk 读完后只 query 一次，最终回答使用当前 KV 窗口，并在每层 decoder 中通过 SSM fusion 从每层 4-direction hidden state 投影出的少量 memory tokens 读取窗口外压缩历史。

注意：这里的 SSM 压缩对象是 decoder attention KV，不是视觉 hidden tokens。当前实现不再维护 `SSM_MAX_MEMORY_LEN` 个历史 memory buffer；历史信息保存在每层 4 个方向的 recurrent hidden state 中。`SSM_MAX_MEMORY_LEN` 参数保留为脚本兼容项，但在当前 KV-CSMS 路径中不参与计算。

常用覆盖参数：

```bash
NUM_FRAMES=-1
MIV_TOKEN_LEN=64
SI_TOKEN_LEN=729
SENSORY_WINDOW_SIZE=140
SLIDING_WINDOW_STRIDE=1
SENSORY_WINDOW_MAX_TOKENS=0
STREAM_VISUAL_MICRO_BATCH_SIZE=1
STREAM_QUERY_MODE=chunk
SSM_D_STATE=64
# SSM_MAX_MEMORY_LEN is kept for CLI compatibility but ignored by current KV-CSMS.
SSM_MAX_MEMORY_LEN=256
SSM_FUSION_NUM_HEADS=8
SSM_FUSION_BOTTLENECK=256
SSM_LAYER_SHARING=group4
```

注意：当前 SSM fusion 新增的参数是随机初始化的机制参数；如果要追求任务效果，需要针对该机制继续训练或微调。未训练时更适合先用于链路验证和效率实验。

## 单视频调试

可以只评测数据集中的一个视频：

```bash
SINGLE_VIDEO_PATH=relative/or/absolute/video.mp4 \
MODEL_VARIANT=vl \
bash scripts/vsr_qwen25vl.sh
```

如果 `SINGLE_VIDEO_PATH` 是相对路径，会相对于 `VSI_SUPER_RECALL_ROOT` 解析。

该模式适合先检查视频切片、prompt 格式、显存占用和单样本输出，再运行完整 split。

## 输出与指标

不同模式的默认输出目录会写到启动命令所在目录下，避免和旧实验日志混在一起：

| 模式       | 性能日志目录                    | lmms-eval 输出目录                |
| ---------- | ------------------------------- | --------------------------------- |
| `vl`     | `./vsr_eval_runs/perf_vl`  | `./vsr_eval_runs/output_vl`  |
| `sw`     | `./vsr_eval_runs/perf_sw`  | `./vsr_eval_runs/output_sw`  |
| `ssm`    | `./vsr_eval_runs/perf_ssm` | `./vsr_eval_runs/output_ssm` |

每次运行会生成：

| 文件            | 内容                                                                             |
| --------------- | -------------------------------------------------------------------------------- |
| `summary.txt` | 任务名、run id、耗时、样本数、吞吐、输出路径、日志路径、切片清理状态、退出状态。 |
| `eval.log`    | 完整的 lmms-eval stdout/stderr。                                                 |
| `gpu.csv`     | 定时采样的 GPU 利用率、显存和功耗。                                              |
| `metrics.csv` | 每个`LOG_ROOT` 下累积保存的耗时和吞吐汇总。                                    |

效率记录只关注运行性能，不记录任务准确率。

## 当前效率记录

以下结果整理自 `/home/ZhangHuayu/Workspace/cambrian-s/lmms-eval/scripts/logs/vsr_eval_runs`。本表只记录 `samples=60` 且 `exit_status=0` 的完整运行；`samples=0` 的失败或中断记录不纳入对比。

| 模式 | run_id | 样本数 | 总耗时(s) | 平均耗时(s/sample) | 吞吐(samples/s) | 单卡峰值显存(GiB) | 各卡峰值均值(GiB) | 8 卡峰值总和(GiB) | 平均 GPU 利用率(%) |
| ---- | ------ | ------ | --------- | ------------------ | --------------- | ----------------- | ----------------- | ----------------- | ------------------ |
| `vl` | `20260714_144732` | 60 | 2061 | 34.350 | 0.029112 | 41.868 | 30.800 | 246.403 | 60.01 |
| `sw` | `20260714_172805` | 60 | 1118 | 18.633 | 0.053667 | 43.778 | 26.217 | 209.736 | 49.61 |
| `ssm` | `20260714_185813` | 60 | 1775 | 29.583 | 0.033803 | 39.513 | 28.223 | 225.785 | 33.75 |

指标含义：

- `总耗时(s)`：本次完整评测的 wall-clock 时间，来自 `summary.txt` 的 `seconds`。
- `平均耗时(s/sample)`：`seconds / samples`，越小表示单样本处理越快。
- `吞吐(samples/s)`：`samples / seconds`，越大表示整体吞吐越高。
- `单卡峰值显存(GiB)`：所有 GPU 中观测到的最大单卡 `memory.used` 峰值，用于判断是否接近 OOM。
- `各卡峰值均值(GiB)`：每张 GPU 各自显存峰值的平均，用于比较典型单卡压力。
- `8 卡峰值总和(GiB)`：8 张 GPU 峰值显存相加，用于粗略比较总显存占用。
- `平均 GPU 利用率(%)`：`gpu.csv` 中 `utilization.gpu` 的平均值，用于观察运行期间 GPU 计算饱和度。

从当前记录看，`sw` 在本轮 SimpleStream 逻辑下比 `vl` 更快；旧表中的 `ssm` 结果来自上一版机制，当前 KV-CSMS 改动后需要重新评测。当前 `ssm` 的 fusion 参数未经过任务训练，本节只做效率链路记录。

## 对比三种机制的推荐模板

为了公平比较不同机制，应保持 GPU、切片长度、重叠、帧数上限一致：

```bash
cd /home/ZhangHuayu/Workspace/cambrian-s/lmms-eval

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CHUNK_SECONDS=150
export CHUNK_OVERLAP_SECONDS=0
export CHUNK_MAX_NUM_FRAMES=150
export CHUNK_ENCODE_MODE=reencode
export CHUNK_PER_VIDEO_FLOW=1
export CHUNK_TEMP_MODE=1
export KEEP_CHUNKS=0

MODEL_VARIANT=vl bash scripts/vsr_qwen25vl.sh
MODEL_VARIANT=sw bash scripts/vsr_qwen25vl.sh
MODEL_VARIANT=ssm bash scripts/vsr_qwen25vl.sh
```

运行结束后主要比较：

- `summary.txt`：耗时、吞吐、样本数和退出状态。
- `metrics.csv`：多次运行的性能汇总。
- 各 `output_*` 目录：lmms-eval 的预测样本与原始输出。
- `gpu.csv`：GPU 利用率、显存峰值和功耗。

## 常见问题

`current transformers lacks Qwen2_5_VLForConditionalGeneration`

当前环境的 `transformers` 不支持 Qwen2.5-VL。切换到正确环境，或升级 `transformers`。

`lmms_eval is not importable in current PYTHON_BIN`

当前 Python 找不到本仓库。可在仓库中执行 `pip install -e .`，或指定 Python：

```bash
PYTHON_BIN=/path/to/python bash scripts/vsr_qwen25vl.sh
```

`vl` 模式显存不足：

- 保持切片开启。
- 降低 `CHUNK_SECONDS`、`CHUNK_MAX_NUM_FRAMES`、`MAX_NUM_FRAMES` 或 `MAX_PIXELS`。
- 使用 `ATTN_IMPLEMENTATION=sdpa`，或当前环境支持的更省显存 attention 实现。

滑动窗口没有明显触发丢帧：

- `sw` 模式下，不建议在较大的 `SENSORY_WINDOW_SIZE` 下使用 `NUM_FRAMES=1`。
- 使用 `NUM_FRAMES=-1`，或设置成接近每个 chunk 实际帧数的值。
- 查看 `eval.log` 中打印的 `num_frames`、`sensory_window_size` 和 `sliding_window_stride`。

运行结束后临时切片消失：

当 `CHUNK_TEMP_MODE=1` 且 `KEEP_CHUNKS=0` 时，这是预期行为。调试时可使用：

```bash
KEEP_CHUNKS=1 bash scripts/vsr_qwen25vl.sh
```

`accelerate` 端口冲突：

覆盖主进程端口：

```bash
MAIN_PROCESS_PORT=29600 MODEL_VARIANT=sw bash scripts/vsr_qwen25vl.sh
```
