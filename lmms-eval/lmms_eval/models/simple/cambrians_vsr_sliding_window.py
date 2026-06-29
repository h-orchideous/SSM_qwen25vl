import math
import os
import functools
from datetime import timedelta
from typing import List, Optional, Tuple, Union
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState

import sys

sys.path = ["../"] + sys.path

from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates
from cambrian.model.builder import load_pretrained_model
from cambrian.mm_utils import tokenizer_image_token, get_model_name_from_path, expand2square
from cambrian.model.cambrian_arch import unpad_image

from decord import VideoReader, cpu
from PIL import Image

def process_video_with_decor_vsr(video_file, model_cfg, num_threads=-1):

    if num_threads < 1:
        vr = VideoReader(video_file, ctx=cpu(0))
    else:
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=num_threads)
    frame_idx = list(range(len(vr)))

    video = vr.get_batch(frame_idx).asnumpy()

    vr.seek(0)
    return video, None, None, None


def process_videos_vsr(videos, image_processor, model_cfg, num_threads=-1):
    processor_aux_list = image_processor
    new_videos_aux_list = []
    video_sizes = []

    for video in videos:
        video, _, _, _ = process_video_with_decor_vsr(video, model_cfg, num_threads=num_threads)
        video_sizes.append((video.shape[2], video.shape[1], video.shape[0])) # W, H, T
        video = [Image.fromarray(video[_], mode="RGB") for _ in range(video.shape[0])] # covert to PIL.Image.Image

        video_aux_list = []
        for processor_aux in processor_aux_list:
            video_aux_list.append(processor_aux.preprocess(video, return_tensors='pt')['pixel_values'])

        new_videos_aux_list.append(video_aux_list)

    new_videos_aux_list = [list(batch_video_aux) for batch_video_aux in zip(*new_videos_aux_list)]
    new_videos_aux_list = [torch.stack(video_aux) for video_aux in new_videos_aux_list]

    return new_videos_aux_list, video_sizes, None


from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

def is_video_file(file_path: str) -> bool:
    if isinstance(file_path, Image.Image):
        return False
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in video_extensions

def is_image_file(file_path: str) -> bool:
    if isinstance(file_path, Image.Image):
        return True
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in image_extensions

@functools.lru_cache(None)
def print_once(*args, **kwargs):
    print(*args, **kwargs)

@register_model("cambrians_vsr")
class CambrianS_VSR(lmms):

    def __init__(
        self,
        pretrained: str = "",
        torch_dtype: Optional[Union[str, torch.dtype]] = "float16",
        batch_size: Optional[Union[int, str]] = 1,
        device_map="cuda:0",
        conv_template="qwen_2",
        use_cache=True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        #############################
        video_max_frames: int = -1,
        video_fps: int = 1,
        video_force_sample: bool = False,
        add_time_instruction: bool = False,
        #############################
        miv_token_len: int = 64,
        si_token_len: int = 729,
        image_aspect_ratio: str = "anyres",
        anyres_max_subimages: int = 9,
        #############################
        enable_visual_feature_caching: bool = False,
        sensory_window_size: int = 512, # disable sensory by setting to -1
        sliding_window_size: Optional[int] = None,
        #############################
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and (device_map == "auto" or device_map == "balanced_low_0"):
            raise NotImplementedError("device_map == auto is not supported yet.")
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        self.model_name = get_model_name_from_path(pretrained)

        self.torch_dtype = torch_dtype
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, self.model_name, device_map=self.device_map)
        ssm_env = os.getenv("CAMBRIAN_USE_SSM_MODEL", "0")
        loaded_model_cls = f"{self._model.__class__.__module__}.{self._model.__class__.__name__}"
        eval_logger.info(f"CAMBRIAN_USE_SSM_MODEL={ssm_env}")
        eval_logger.info(f"Loaded model class: {loaded_model_cls}")

        self._model.config.video_max_frames = video_max_frames
        self._model.config.video_fps = video_fps
        self._model.config.video_force_sample = video_force_sample
        self._model.config.add_time_instruction = add_time_instruction
        self._model.config.miv_token_len = miv_token_len
        self._model.config.si_token_len = si_token_len
        self._model.config.image_aspect_ratio = image_aspect_ratio
        self._model.config.anyres_max_subimages = anyres_max_subimages

        eval_logger.info(f"video_max_frames: {video_max_frames}")
        eval_logger.info(f"video_fps: {video_fps}")
        eval_logger.info(f"video_force_sample: {video_force_sample}")
        eval_logger.info(f"add_time_instruction: {add_time_instruction}")
        eval_logger.info(f"miv_token_len: {miv_token_len}")
        eval_logger.info(f"si_token_len: {si_token_len}")
        eval_logger.info(f"image_aspect_ratio: {image_aspect_ratio}")
        eval_logger.info(f"anyres_max_subimages: {anyres_max_subimages}")

        if sliding_window_size is not None:
            sensory_window_size = sliding_window_size

        self.sensory_window_size = sensory_window_size

        eval_logger.info(f"sensory_window_size: {sensory_window_size}")

        self._config = self._model.config

        self.model.eval()
        print(self.model)

        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context

        self.enable_visual_feature_caching = enable_visual_feature_caching

        if self.enable_visual_feature_caching:
            self.cache_dir = os.path.join(".cache", self.pretrained, "visual_features")
            os.makedirs(self.cache_dir, exist_ok=True)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, requests, task_dict, tokenizer, image_processor, model_config, conv_template, pretrained, enable_visual_feature_caching, cache_dir):
                self.requests = requests
                self.task_dict = task_dict
                self.tokenizer = tokenizer
                self.image_processor = image_processor
                self.model_config = model_config
                self.conv_template = conv_template
                self.pretrained = pretrained
                self.enable_visual_feature_caching = enable_visual_feature_caching
                self.cache_dir = cache_dir

            def __len__(self):
                return len(self.requests)

            def __getitem__(self, idx):

                def feature_path_exists(paths):
                    if not self.enable_visual_feature_caching: return False
                    for path in paths:
                        if not os.path.exists(os.path.join(self.cache_dir, path.replace("/", "_") + ".pt")):
                            return False
                    return True

                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = self.requests[idx].args
                visuals = doc_to_visual(self.task_dict[task][split][doc_id])

                if visuals is not None:
                    qs = contexts
                    try:
                        assert len(visuals) == 1
                        assert isinstance(visuals[0], str)
                        assert is_video_file(visuals[0])
                        if not feature_path_exists(visuals):
                            visual_tensors, visual_sizes, _ = process_videos_vsr(visuals, self.image_processor, self.model_config, num_threads=-1)
                            visual_tensors_type = "raw"
                            visual_tensor_paths = [path.replace("/", "_") + ".pt" for path in visuals]
                            visual_tensor_paths = visuals[0].replace("/", "_") + ".pt"
                        else:
                            visual_tensor_paths = visuals[0].replace("/", "_") + ".pt"
                            visual_tensors = torch.load(os.path.join(self.cache_dir, visual_tensor_paths))
                            visual_tensors_type = "feature"
                            visual_sizes = torch.load(os.path.join(self.cache_dir, visual_tensor_paths.replace(".pt", "_size.pt")))
                    except Exception as e:
                        raise e

                    if isinstance(qs, str):
                        if self.model_config.mm_use_im_start_end:
                            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                        else:
                            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                    else:
                        raise NotImplementedError

                else:
                    visual_tensors = None
                    visual_sizes = None
                    qs = contexts

                conv = conv_templates[self.conv_template].copy()

                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
                return input_ids, visual_tensors_type, visual_tensors, visual_sizes, prompt, gen_kwargs, visual_tensor_paths, contexts, doc_id

        dataset = Dataset(requests, self.task_dict, self.tokenizer, self._image_processor, self._config, self.conv_template, self.pretrained, self.enable_visual_feature_caching, self.cache_dir)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0], num_workers=0, pin_memory=True)

        for _, (input_ids, visual_tensors_type, visual_tensors, visual_sizes, cur_prompt, gen_kwargs, visual_tensor_paths, contexts, doc_id) in enumerate(dataloader):

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 16
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            with torch.inference_mode():

                def add_newline_tokens(visual_features):
                    visual_features = torch.cat([visual_features, self.model.model.image_newline[None, None, None, :].expand(*visual_features.size()[:2], 1, -1)], dim=2) # BHWC -> BH(W+1)C
                    visual_features = visual_features.flatten(1, 2).flatten(0, 1) # BHWC -> (BHW)C
                    return visual_features

                if visual_tensors_type == "raw":
                    # extract image features
                    visual_tensors = visual_tensors[0].flatten(0, 1)
                    block_size = 128
                    visual_features = []
                    miv_token_len = self.model.get_model().config.miv_token_len
                    miv_side_len = int(math.sqrt(miv_token_len))

                    for bid in range(math.ceil(visual_tensors.size(0) / block_size)):
                        chunked_visual_features = visual_tensors[bid * block_size : (bid + 1) * block_size].half().to(self._device)
                        chunked_visual_features = self.model.encode_images([chunked_visual_features])[0]
                        chunked_visual_features = self.model.get_model().mm_projector(chunked_visual_features)
                        
                        feature_side_len = int(math.sqrt(chunked_visual_features.size(1)))
                        chunked_visual_features = chunked_visual_features.unflatten(1, (feature_side_len, feature_side_len)).permute(0, 3, 1, 2)
                        if feature_side_len != miv_side_len:
                            chunked_visual_features = torch.nn.functional.interpolate(chunked_visual_features, size=(miv_side_len, miv_side_len), mode="bilinear", align_corners=False)
                            chunked_visual_features = chunked_visual_features.permute(0, 2, 3, 1)

                        visual_features.append(chunked_visual_features)
                    visual_features = torch.cat(visual_features, dim=0)

                    eval_logger.info("Saving visual features to disk...")
                    if self.enable_visual_feature_caching:
                        if isinstance(visual_tensor_paths, list):
                            for _, visual_tensor_path in enumerate(visual_tensor_paths):
                                visual_feature = visual_features[_:_+1].cpu()
                                if not os.path.exists(self.cache_dir):
                                    os.makedirs(self.cache_dir)
                                torch.save(visual_feature, os.path.join(self.cache_dir, visual_tensor_path))
                        elif isinstance(visual_tensor_paths, str):
                            visual_feature = visual_features.cpu()
                            if not os.path.exists(self.cache_dir):
                                os.makedirs(self.cache_dir)
                            torch.save(visual_feature, os.path.join(self.cache_dir, visual_tensor_paths))
                            torch.save(visual_sizes, os.path.join(self.cache_dir, visual_tensor_paths.replace(".pt", "_size.pt")))
                        else:
                            raise NotImplementedError
                elif visual_tensors_type == "feature":
                    # skip visual encoder
                    visual_tensors = visual_tensors
                    visual_features = visual_tensors.to(self._device)

                else:
                    raise NotImplementedError

                visual_features = unpad_image(visual_features, visual_sizes[0][:2])
                from qwen2_monkey_patch import Qwen2SdpaAttention
                for layer in self.model.model.layers:
                    layer.self_attn.__class__ = Qwen2SdpaAttention
                from qwen2_monkey_patch import cambrian_qwen2_forward
                from cambrian.model.language_model.cambrian_qwen2 import CambrianQwenModel
                CambrianQwenModel.forward = cambrian_qwen2_forward

                input_ids = input_ids.to(self._device)
                assert input_ids.size(0) == 1
                pre_img_tokens = input_ids[:, :torch.where(input_ids[0]==-200)[0][0]]
                pre_img_embeds = self.model.get_input_embeddings()(pre_img_tokens)
                runtime_kv_cache = [{"key_states": [], "value_states": [], "lengths": []} for _ in range(self.model.config.num_hidden_layers)]

                out = self.model(
                    input_ids=None,
                    inputs_embeds=pre_img_embeds,
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=None,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )

                for layer_idx, layer_cache in enumerate(out.past_key_values):
                    if hasattr(layer_cache, "to_legacy_cache"):
                        layer_cache = layer_cache.to_legacy_cache()

                    if len(layer_cache) == 1 and isinstance(layer_cache[0], tuple):
                        layer_cache = layer_cache[0]

                    if len(layer_cache) != 2:
                        raise ValueError(f"Unexpected cache structure for layer {layer_idx}: {type(layer_cache)} with length {len(layer_cache)}")

                    key_states, value_states = layer_cache
                    runtime_kv_cache[layer_idx]["key_states"].append(key_states)
                    runtime_kv_cache[layer_idx]["value_states"].append(value_states)
                    runtime_kv_cache[layer_idx]["lengths"].append(key_states.size(2))

                for frame_idx in range(visual_features.size(0)):
                    past_key_values = []

                    for layer_idx, layer_wise_runtime_cache in enumerate(runtime_kv_cache):
                        past_key_values.append((
                            torch.cat(layer_wise_runtime_cache["key_states"], dim=2),
                            torch.cat(layer_wise_runtime_cache["value_states"], dim=2),
                        ))

                    frame_feature = visual_features[frame_idx:frame_idx+1]

                    frame_feature = add_newline_tokens(frame_feature).unsqueeze(0)
                    input_embeds = frame_feature

                    out = self.model(
                        input_ids=None,
                        inputs_embeds=input_embeds,
                        attention_mask=None,
                        position_ids=None,
                        use_cache=True,
                        return_dict=True,
                        past_key_values=past_key_values,
                        output_attentions=False,
                        output_hidden_states=True,
                    )

                    for layer_idx, (layer_wise_past_key_values, layer_wise_output_cache) in enumerate(zip(past_key_values, out.past_key_values)):
                        input_seq_len = layer_wise_past_key_values[0].size(2)
                        runtime_kv_cache[layer_idx]["key_states"].append(layer_wise_output_cache[0][..., input_seq_len:, :].clone())
                        runtime_kv_cache[layer_idx]["value_states"].append(layer_wise_output_cache[1][..., input_seq_len:, :].clone())
                        runtime_kv_cache[layer_idx]["lengths"].append(runtime_kv_cache[layer_idx]["key_states"][-1].size(2))
                        if self.sensory_window_size > 0 and len(runtime_kv_cache[layer_idx]["key_states"]) > self.sensory_window_size + 1:
                            runtime_kv_cache[layer_idx]["key_states"].pop(1)
                            runtime_kv_cache[layer_idx]["value_states"].pop(1)
                            runtime_kv_cache[layer_idx]["lengths"].pop(1)
                past_key_values = []

                for layer_idx in range(self.model.config.num_hidden_layers):
                    layer_keys_list = [runtime_kv_cache[layer_idx]["key_states"][0]]
                    layer_values_list = [runtime_kv_cache[layer_idx]["value_states"][0]]

                    if len(runtime_kv_cache[layer_idx]["key_states"]) > 1:
                        layer_keys_list.extend(runtime_kv_cache[layer_idx]["key_states"][1:])
                        layer_values_list.extend(runtime_kv_cache[layer_idx]["value_states"][1:])

                    layer_keys = torch.cat(layer_keys_list, dim=2)
                    layer_values = torch.cat(layer_values_list, dim=2)
                    past_key_values.append((layer_keys, layer_values))

                post_img_tokens = input_ids[:, torch.where(input_ids[0]==-200)[0][0]+1:]
                post_img_embeds = self.model.get_input_embeddings()(post_img_tokens)

                out = self.model(
                    input_ids=None,
                    inputs_embeds=post_img_embeds,
                    attention_mask=None,
                    position_ids=None,
                    use_cache=True,
                    return_dict=True,
                    past_key_values=past_key_values,
                    output_attentions=False,
                    output_hidden_states=True,
                )
                past_key_values = out.past_key_values

                logits = out.logits[:, -1, :]
                pred = logits.argmax(dim=-1)
                output_ids = torch.cat([torch.zeros_like(pred)[:, None].long().fill_(self._tokenizer.pad_token_id), pred[:, None]], dim=1)

                for _ in range(gen_kwargs["max_new_tokens"] - 1):
                    if pred == self._tokenizer.eos_token_id:
                        break
                    out = self.model(
                        input_ids=output_ids[:, -1:],
                        inputs_embeds=None,
                        attention_mask=None,
                        position_ids=None,
                        use_cache=True,
                        return_dict=True,
                        past_key_values=past_key_values,
                        output_attentions=False,
                        output_hidden_states=True,
                    )
                    past_key_values = out.past_key_values
                    logits = out.logits[:, -1, :]
                    
                    # 1.1 repetation penalty by default
                    score = torch.gather(logits, 1, output_ids)
                    score = torch.where(score < 0, score * 1.1, score / 1.1)
                    logits.scatter_(1, output_ids, score)
                    
                    pred = logits.argmax(dim=-1)
                    output_ids = torch.cat([output_ids, pred[:, None]], dim=-1)

            outputs = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            eval_logger.info(f"Question: {cur_prompt}")
            eval_logger.info(f"Answer: {outputs}")
            res.append(outputs)
            pbar.update(1)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError
