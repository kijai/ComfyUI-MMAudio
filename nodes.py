import os
import torch
import json
from torchvision.transforms import v2
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file

script_directory = os.path.dirname(os.path.abspath(__file__))

if not "mmaudio" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("mmaudio", os.path.join(folder_paths.models_dir, "mmaudio"))

from .mmaudio.eval_utils import generate
from .mmaudio.model.flow_matching import FlowMatching
from .mmaudio.model.networks import MMAudio
from .mmaudio.model.utils.features_utils import FeaturesUtils
from .mmaudio.model.sequence_config import (CONFIG_16K, CONFIG_44K, SequenceConfig)
from .mmaudio.ext.bigvgan_v2.bigvgan import BigVGAN as BigVGANv2
from .mmaudio.ext.synchformer import Synchformer
from .mmaudio.ext.autoencoder import AutoEncoderModule
from open_clip import CLIP

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def process_video_tensor(
    video_tensor: torch.Tensor,
    duration_sec: float,
    target_clip: int | None = None,
    target_sync: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Prepare visual tensors for MMAudio.

    - Uniformly resamples over the FULL input span (no front-slice).
    - Uses exact target counts for CLIP tokens and SYNC TOKENS expected by the model.
    - IMPORTANT (Synchformer): to GET `target_sync` TOKENS OUT, we must feed
      `target_sync + 8` RAW FRAMES in. Reason:
        * segments of 16 frames, stride 8  -> num_segments = floor((N-16)/8) + 1
        * downsample_rate 2                -> tokens = num_segments * 16 / 2 = num_segments * 8
        * algebra => tokens ≈ N − 8  (so add 8 to N to hit tokens exactly)
    - Keeps `duration_sec` unchanged. Any upsampling is handled via duplicated indices.

    Args:
        video_tensor: (T, H, W, C) float in [0,1], typically on CUDA (ComfyUI IMAGE).
        duration_sec: Requested audio duration (returned unchanged).
        target_clip:  Desired CLIP token length (e.g., 64 for 8s, 80 for 10s).
        target_sync:  Desired SYNC token length (e.g., 192 for 8s, 242 for 10s).

    Returns:
        clip_frames: (T_clip, 3, 384, 384) float32 in [0,1]
        sync_frames: (T_sync_raw, 3, 224, 224) float32 normalized to [-1,1]
                     NOTE: T_sync_raw == target_sync + 8 (by design).
        duration_sec: unchanged
    """
    _CLIP_SIZE = 384
    _CLIP_FPS  = 8.0

    _SYNC_SIZE = 224
    _SYNC_FPS  = 25.0

    # Synchformer sampling constants from upstream config
    _SYNC_NUM_FRAMES_PER_SEG = 16
    _SYNC_STEP_SIZE          = 8
    _SYNC_OFFSET             = _SYNC_NUM_FRAMES_PER_SEG - _SYNC_STEP_SIZE  # = 8

    # CPU/PIL transforms
    clip_transform = v2.Compose([
        v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToPILImage(),
        v2.ToTensor(),
        v2.ConvertImageDtype(torch.float32),
    ])

    sync_transform = v2.Compose([
        v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_SYNC_SIZE),
        v2.ToPILImage(),
        v2.ToTensor(),
        v2.ConvertImageDtype(torch.float32),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    total_frames = int(video_tensor.shape[0])
    if total_frames <= 0:
        raise ValueError("process_video_tensor: input video_tensor has zero frames")

    # Defaults from duration, if explicit targets aren't provided
    if target_clip is None:
        target_clip = max(1, int(_CLIP_FPS * duration_sec))
    if target_sync is None:
        # NOTE: this is the TOKEN count; we will fetch raw frames = target_sync + 8
        target_sync = max(1, int(_SYNC_FPS * duration_sec) - _SYNC_OFFSET)

    # EVENLY spaced indices helper (same device as source to satisfy index_select)
    def uniform_indices(n_src: int, n_tgt: int, device: torch.device) -> torch.Tensor:
        if n_src == 1:
            return torch.zeros(n_tgt, dtype=torch.long, device=device)
        idx = torch.linspace(0, n_src - 1, steps=n_tgt, device=device)
        idx = torch.floor(idx).to(torch.long).clamp_(0, n_src - 1)
        return idx

    # --- CLIP stream: target == token count == raw frame count ---
    clip_idx = uniform_indices(total_frames, target_clip, video_tensor.device)
    clip_frames = video_tensor.index_select(0, clip_idx)

    # --- SYNC stream: need RAW_FRAMES = target_sync_tokens + 8 ---
    sync_raw_needed = max(_SYNC_NUM_FRAMES_PER_SEG, int(target_sync) + _SYNC_OFFSET)
    sync_idx = uniform_indices(total_frames, sync_raw_needed, video_tensor.device)
    sync_frames = video_tensor.index_select(0, sync_idx)

    # Move to CPU, CHW, then transform
    clip_frames = clip_frames.cpu().permute(0, 3, 1, 2)
    sync_frames = sync_frames.cpu().permute(0, 3, 1, 2)

    clip_frames = torch.stack([clip_transform(frame) for frame in clip_frames])
    sync_frames = torch.stack([sync_transform(frame) for frame in sync_frames])

    return clip_frames, sync_frames, duration_sec

#region Model loading
class MMAudioModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mmaudio_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from the 'ComfyUI/models/mmaudio' -folder",}),
            
            "base_precision": (["fp16", "fp32", "bf16"], {"default": "fp16"}),
            },
        }

    RETURN_TYPES = ("MMAUDIO_MODEL",)
    RETURN_NAMES = ("mmaudio_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "MMAudio"

    def loadmodel(self, mmaudio_model, base_precision):
       

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
       
        mm.soft_empty_cache()

        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[base_precision]

        mmaudio_model_path = folder_paths.get_full_path_or_raise("mmaudio", mmaudio_model)
        mmaudio_sd = load_torch_file(mmaudio_model_path, device=offload_device)

        with init_empty_weights():
            # small
            if mmaudio_sd["audio_input_proj.0.bias"].shape[0] == 448:
                num_heads = 7
                model = MMAudio(
                        latent_dim=40,
                        clip_dim=1024,
                        sync_dim=768,
                        text_dim=1024,
                        hidden_dim=64 * num_heads,
                        depth=12,
                        fused_depth=8,
                        num_heads=num_heads,
                        latent_seq_len=345,
                        clip_seq_len=64,
                        sync_seq_len=192
                        )
            # large
            elif mmaudio_sd["audio_input_proj.0.bias"].shape[0] == 896:
                num_heads = 14
                model = MMAudio(latent_dim=40,
                        clip_dim=1024,
                        sync_dim=768,
                        text_dim=1024,
                        hidden_dim=64 * num_heads,
                        depth=21,
                        fused_depth=14,
                        num_heads=num_heads,
                        latent_seq_len=345,
                        clip_seq_len=64,
                        sync_seq_len=192,
                        v2=mmaudio_sd["t_embed.mlp.0.weight"].shape[1] == 896
                        )
        model = model.eval()
        for name, param in model.named_parameters():
            # Set tensor to offload_device to not consume VRAM early in workflow
            set_module_tensor_to_device(model, name, device=offload_device, dtype=base_dtype, value=mmaudio_sd[name])
        del mmaudio_sd
        log.info(f'Loaded MMAudio model weights from {mmaudio_model_path}')
        if "44" in mmaudio_model:
            model.seq_cfg = CONFIG_44K
        elif "16" in mmaudio_model:
            model.seq_cfg = CONFIG_16K
        
        return (model,)
    
#region Features Utils
class MMAudioVoCoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vocoder_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),  
            },
        }

    RETURN_TYPES = ("VOCODER_MODEL",)
    RETURN_NAMES = ("mmaudio_vocoder", )
    FUNCTION = "loadmodel"
    CATEGORY = "MMAudio"

    def loadmodel(self, vocoder_model):
        from .mmaudio.ext.bigvgan import BigVGAN
        vocoder_model_path = folder_paths.get_full_path_or_raise("mmaudio", vocoder_model)
        vocoder_model = BigVGAN.from_pretrained(vocoder_model_path).eval()
        return (vocoder_model_path,)
        
class MMAudioFeatureUtilsLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
                "synchformer_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
                "clip_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
            },
            "optional": {
              "bigvgan_vocoder_model": ("VOCODER_MODEL", {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
                "mode": (["16k", "44k"], {"default": "44k"}),
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "fp16"}
                ),
            }
        }

    RETURN_TYPES = ("MMAUDIO_FEATUREUTILS",)
    RETURN_NAMES = ("mmaudio_featureutils", )
    FUNCTION = "loadmodel"
    CATEGORY = "MMAudio"

    def loadmodel(self, vae_model, precision, synchformer_model, clip_model, mode, bigvgan_vocoder_model=None):
        
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        #synchformer
        synchformer_path = folder_paths.get_full_path_or_raise("mmaudio", synchformer_model)
        synchformer_sd = load_torch_file(synchformer_path, device=offload_device)
        with init_empty_weights():
            synchformer = Synchformer().eval()
        
        for name, param in synchformer.named_parameters():
            # Set tensor to device
            set_module_tensor_to_device(synchformer, name, device=device, dtype=dtype, value=synchformer_sd[name])

        #vae
        download_path = folder_paths.get_folder_paths("mmaudio")[0]

        nvidia_bigvgan_vocoder_path = os.path.join(download_path, "nvidia", "bigvgan_v2_44khz_128band_512x")
        if mode == "44k":
            if not os.path.exists(nvidia_bigvgan_vocoder_path):
                log.info(f"Downloading nvidia bigvgan vocoder model to: {nvidia_bigvgan_vocoder_path}")
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id="nvidia/bigvgan_v2_44khz_128band_512x",
                    ignore_patterns=["*3m*",],
                    local_dir=nvidia_bigvgan_vocoder_path,
                    local_dir_use_symlinks=False,
                )
            
            bigvgan_vocoder = BigVGANv2.from_pretrained(nvidia_bigvgan_vocoder_path).eval().to(device=device, dtype=dtype)
        else:
            assert bigvgan_vocoder_model is not None, "bigvgan_vocoder_model must be provided for 16k mode"
            bigvgan_vocoder = bigvgan_vocoder_model
        
        vae_path = folder_paths.get_full_path_or_raise("mmaudio", vae_model)
        vae_sd = load_torch_file(vae_path, device=offload_device)
        vae = AutoEncoderModule(
            vae_state_dict=vae_sd,
            bigvgan_vocoder=bigvgan_vocoder,
            mode=mode
            )
        vae = vae.eval().to(device=device, dtype=dtype)

        #clip
       
        clip_model_path = folder_paths.get_full_path_or_raise("mmaudio", clip_model)
        clip_config_path = os.path.join(script_directory, "configs", "DFN5B-CLIP-ViT-H-14-384.json")
        with open(clip_config_path) as f:
             clip_config = json.load(f)
            
        with init_empty_weights():
            try:
                clip_model = CLIP(**clip_config["model_cfg"]).eval()
            except:
                # for some open-clip versions
                clip_config["model_cfg"]["nonscalar_logit_scale"] = True
                clip_model = CLIP(**clip_config["model_cfg"]).eval()

        clip_sd = load_torch_file(os.path.join(clip_model_path), device=offload_device)
        for name, param in clip_model.named_parameters():
            set_module_tensor_to_device(clip_model, name, device=device, dtype=dtype, value=clip_sd[name])
        clip_model.to(device=device, dtype=dtype)

        #clip_model = create_model_from_pretrained("hf-hub:apple/DFN5B-CLIP-ViT-H-14-384", return_transform=False)
        
       
        feature_utils = FeaturesUtils(vae=vae,
                                  synchformer=synchformer,
                                  enable_conditions=True,
                                  clip_model=clip_model)
        return (feature_utils,)

#region sampling
class MMAudioSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mmaudio_model": ("MMAUDIO_MODEL",),
                "feature_utils": ("MMAUDIO_FEATUREUTILS",),
                "duration": ("FLOAT", {"default": 8, "step": 0.01, "tooltip": "Duration of the audio in seconds"}),
                "steps": ("INT", {"default": 25, "step": 1, "tooltip": "Number of steps to interpolate"}),
                "cfg": ("FLOAT", {"default": 4.5, "step": 0.1, "tooltip": "Strength of the conditioning"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt": ("STRING", {"default": "", "multiline": True} ),
                "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
                "mask_away_clip": ("BOOLEAN", {"default": False, "tooltip": "If true, the clip video will be masked away"}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "If true, the model will be offloaded to the offload device"}),
            },
            "optional": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio", )
    FUNCTION = "sample"
    CATEGORY = "MMAudio"

    def sample(self, mmaudio_model, seed, feature_utils, duration, steps, cfg, prompt, negative_prompt, mask_away_clip, force_offload, images=None):
        """
        - Uses seq_cfg to derive the *token* lengths the model expects.
        - Resamples frames so CLIP tokens == seq_cfg.clip_seq_len,
        and SYNC TOKENS == seq_cfg.sync_seq_len (by feeding +8 raw frames).
        - Calls update_seq_lengths with the SAME lengths to avoid asserts.
        - JIT-moves models to GPU; offloads after.
        """
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        rng = torch.Generator(device=device).manual_seed(seed)

        # 1) Fix duration in config; derive target token counts from cfg
        seq_cfg: SequenceConfig = mmaudio_model.seq_cfg
        seq_cfg.duration = float(duration)

        target_clip_tokens = int(seq_cfg.clip_seq_len)   # e.g., 64 for 8s, 80 for 10s
        target_sync_tokens = int(seq_cfg.sync_seq_len)   # e.g., 192 for 8s, 242 for 10s

        # 2) Prepare visual inputs
        if images is not None:
            images = images.to(device=device, non_blocking=True)

            # Resample across the FULL span; note sync path feeds +8 raw frames
            clip_frames, sync_frames, duration = process_video_tensor(
                images, duration,
                target_clip=target_clip_tokens,
                target_sync=target_sync_tokens,
            )
            log.info(
                f"Prepared visual streams: "
                f"CLIP(raw)={clip_frames.shape[0]} (tokens={target_clip_tokens}) @8fps, "
                f"SYNC(raw)={sync_frames.shape[0]} (tokens={target_sync_tokens}) @25fps, "
                f"duration={duration:.3f}s"
            )

            if mask_away_clip:
                clip_frames = None
            else:
                clip_frames = clip_frames.unsqueeze(0)  # (1, T_clip_raw, C, H, W)
            sync_frames = sync_frames.unsqueeze(0)      # (1, T_sync_raw, C, H, W)
        else:
            clip_frames = None
            sync_frames = None

        # 3) Update model to EXPECT these token lengths
        mmaudio_model.update_seq_lengths(
            seq_cfg.latent_seq_len,      # audio latents (comes from audio cfg/rate)
            target_clip_tokens,          # CLIP tokens
            target_sync_tokens           # SYNC tokens
        )

        # 4) Scheduler + JIT device placement
        scheduler = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=steps)
        feature_utils.to(device)
        mmaudio_model.to(device)

        # 5) Generate
        audios = generate(
            clip_frames,
            sync_frames,
            [prompt],
            negative_text=[negative_prompt],
            feature_utils=feature_utils,
            net=mmaudio_model,
            fm=scheduler,
            rng=rng,
            cfg_strength=cfg,
        )

        # 6) Offload if requested
        if force_offload:
            mmaudio_model.to(offload_device)
            feature_utils.to(offload_device)
            mm.soft_empty_cache()

        waveform = audios.float().cpu()
        return ({"waveform": waveform, "sample_rate": 44100},)
        
NODE_CLASS_MAPPINGS = {
    "MMAudioModelLoader": MMAudioModelLoader,
    "MMAudioFeatureUtilsLoader": MMAudioFeatureUtilsLoader,
    "MMAudioSampler": MMAudioSampler,
    "MMAudioVoCoderLoader": MMAudioVoCoderLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MMAudioModelLoader": "MMAudio ModelLoader",
    "MMAudioFeatureUtilsLoader": "MMAudio FeatureUtilsLoader",
    "MMAudioSampler": "MMAudio Sampler",
    "MMAudioVoCoderLoader": "MMAudio VoCoderLoader",
    }
