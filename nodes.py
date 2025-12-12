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

def process_video_tensor(video_tensor: torch.Tensor, duration_sec: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    _CLIP_SIZE = 384
    _CLIP_FPS = 8.0

    _SYNC_SIZE = 224
    _SYNC_FPS = 25.0

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

    # Assuming video_tensor is in the shape (frames, height, width, channels)
    total_frames = video_tensor.shape[0]
    clip_frames_count = int(_CLIP_FPS * duration_sec)
    sync_frames_count = int(_SYNC_FPS * duration_sec)

    # Adjust duration if there are not enough frames
    if total_frames < clip_frames_count:
        log.warning(f'Clip video is too short: {total_frames / _CLIP_FPS:.2f} < {duration_sec:.2f}')
        clip_frames_count = total_frames
        duration_sec = total_frames / _CLIP_FPS

    if total_frames < sync_frames_count:
        log.warning(f'Sync video is too short: {total_frames / _SYNC_FPS:.2f} < {duration_sec:.2f}, truncating to {total_frames / _SYNC_FPS:.2f} sec')
        sync_frames_count = total_frames
        duration_sec = total_frames / _SYNC_FPS

    clip_frames = video_tensor[:clip_frames_count]
    sync_frames = video_tensor[:sync_frames_count]

    clip_frames = clip_frames.permute(0, 3, 1, 2)
    sync_frames = sync_frames.permute(0, 3, 1, 2)

    clip_frames = torch.stack([clip_transform(frame) for frame in clip_frames])
    sync_frames = torch.stack([sync_transform(frame) for frame in sync_frames])

    clip_length_sec = clip_frames.shape[0] / _CLIP_FPS
    sync_length_sec = sync_frames.shape[0] / _SYNC_FPS

    # if clip_length_sec < duration_sec:
    #     log.warning(f'Clip video is too short: {clip_length_sec:.2f} < {duration_sec:.2f}')
    #     log.warning(f'Truncating to {clip_length_sec:.2f} sec')
    #     duration_sec = clip_length_sec

    # if sync_length_sec < duration_sec:
    #     log.warning(f'Sync video is too short: {sync_length_sec:.2f} < {duration_sec:.2f}')
    #     log.warning(f'Truncating to {sync_length_sec:.2f} sec')
    #     duration_sec = sync_length_sec

    clip_frames = clip_frames[:int(_CLIP_FPS * duration_sec)]
    sync_frames = sync_frames[:int(_SYNC_FPS * duration_sec)]

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
            # Set tensor to device
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
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        seq_cfg = mmaudio_model.seq_cfg
       
        if images is not None:
            images = images.to(device=device)
            clip_frames, sync_frames, duration = process_video_tensor(images, duration)
            print("clip_frames", clip_frames.shape, "sync_frames", sync_frames.shape, "duration", duration)
            if mask_away_clip:
                clip_frames = None
            else:
                clip_frames = clip_frames.unsqueeze(0)
            sync_frames = sync_frames.unsqueeze(0)
        else:
            clip_frames = None
            sync_frames = None
        
        seq_cfg.duration = duration
        mmaudio_model.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

        scheduler = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=steps)
        feature_utils.to(device)
        mmaudio_model.to(device)
        audios = generate(clip_frames,
                      sync_frames, [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net=mmaudio_model,
                      fm=scheduler,
                      rng=rng,
                      cfg_strength=cfg)
        if force_offload:
            mmaudio_model.to(offload_device)
            feature_utils.to(offload_device)
            mm.soft_empty_cache()
        waveform = audios.float().cpu()
        #torchaudio.save("test.wav", waveform, 44100)
        audio = {
            "waveform": waveform,
            "sample_rate": 44100
        }

        return (audio,)
        
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
