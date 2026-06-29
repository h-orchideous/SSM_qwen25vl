import os
import json
import sys

DATA_PATH = os.environ.get('DATA_PATH', '/data1/ZhangHuayu/datasets/VSI-Train-10k/vsi_train_10k.jsonl')
SI_TOKEN_LEN = int(os.environ.get('SI_TOKEN_LEN', os.environ.get('si_token_len', '729')))
VIDEO_MAX_FRAMES = int(os.environ.get('VIDEO_MAX_FRAMES', os.environ.get('video_max_frames', '512')))
MODEL_MAX_LENGTH = int(os.environ.get('MODEL_MAX_LENGTH', os.environ.get('MODEL_MAX_LENGTH', '2048')))
SSM_MAX_MEMORY_LEN = int(os.environ.get('SSM_MAX_MEMORY_LEN', os.environ.get('ssm_max_memory_len', '256')))

print(f"[PROBE] DATA_PATH={DATA_PATH}")
print(f"[PROBE] SI_TOKEN_LEN={SI_TOKEN_LEN}, VIDEO_MAX_FRAMES={VIDEO_MAX_FRAMES}, MODEL_MAX_LENGTH={MODEL_MAX_LENGTH}, SSM_MAX_MEMORY_LEN={SSM_MAX_MEMORY_LEN}")

# Try to read first JSONL record to detect modality
if not os.path.exists(DATA_PATH):
    print(f"[WARNING] DATA_PATH not found: {DATA_PATH}. Exiting probe with computed estimates.")
    sys.exit(0)

with open(DATA_PATH, 'r') as f:
    first_line = None
    for line in f:
        line=line.strip()
        if line:
            first_line = line
            break

if first_line is None:
    print('[WARNING] DATA_PATH seems empty. Exiting.')
    sys.exit(0)

try:
    rec = json.loads(first_line)
except Exception as e:
    print('[WARNING] Failed to parse first jsonl line:', e)
    rec = {}

# Heuristics to detect if record is video or image
is_video = False
frame_count = None
if 'video' in rec or 'video_path' in rec or 'video_frames' in rec:
    is_video = True
    frame_count = rec.get('video_frames') or rec.get('num_frames') or None
elif 'image' in rec or 'image_path' in rec:
    is_video = False

print('[PROBE] First record keys:', list(rec.keys())[:20])
if is_video:
    print('[PROBE] Record appears to be video. frame_count field (if present):', frame_count)
else:
    print('[PROBE] Record appears to be image (or not clearly a video).')

# Compute estimates
KV_per_frame = SI_TOKEN_LEN
total_positions_for_frames = KV_per_frame * VIDEO_MAX_FRAMES
print(f"[PROBE] Estimated KV_per_frame = {KV_per_frame}")
print(f"[PROBE] If using VIDEO_MAX_FRAMES={VIDEO_MAX_FRAMES}, total KV positions = {total_positions_for_frames}")

# Recommendations
if MODEL_MAX_LENGTH < total_positions_for_frames:
    print('[RECOMMEND] model_max_length is too small to contain all frames without eviction/compression.')
    approx_frames_fit = MODEL_MAX_LENGTH // KV_per_frame
    print(f"[RECOMMEND] With model_max_length={MODEL_MAX_LENGTH} you can fit approx {approx_frames_fit} full frames (ignoring other tokens).")
else:
    print('[RECOMMEND] model_max_length can hold all frames without truncation (in principle).')

recommended_ssm = int(os.environ.get('DESIRED_SSM_FRAMES', '0'))
if recommended_ssm > 0:
    rec_ssm = recommended_ssm * KV_per_frame
    print(f"[RECOMMEND] Based on DESIRED_SSM_FRAMES={recommended_ssm}, recommended SSM_MAX_MEMORY_LEN={rec_ssm}")
else:
    print(f"[RECOMMEND] Current SSM_MAX_MEMORY_LEN={SSM_MAX_MEMORY_LEN}. To keep N frames in SSM, set DESIRED_SSM_FRAMES and enable SSM_AUTO_CALC in the script to compute it.")

print('\n[PROBE] Summary:')
print(f"  - KV_per_frame = {KV_per_frame}")
print(f"  - video_max_frames = {VIDEO_MAX_FRAMES}")
print(f"  - total_positions = {total_positions_for_frames}")
print(f"  - model_max_length = {MODEL_MAX_LENGTH}")
print(f"  - ssm_max_memory_len(current) = {SSM_MAX_MEMORY_LEN}")

print('\n[PROBE] Done.')
