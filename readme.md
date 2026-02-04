# ComfyUI nodes to use [MMAudio](https://github.com/hkchengrex/MMAudio)

## WIP WIP WIP

https://github.com/user-attachments/assets/9515c0f6-cc5d-4dfe-a642-f841a1a2dba5

# Installation
Clone this repo into custom_nodes folder.

Install dependencies: pip install -r requirements.txt or if you use the portable install, run this in ComfyUI_windows_portable -folder:

python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-MMAudio\requirements.txt


Models are loaded from `ComfyUI/models/mmaudio`

Safetensors available here:

https://huggingface.co/Kijai/MMAudio_safetensors/tree/main

Nvidia bigvganv2 (used with 44k mode)

https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x

is autodownloaded to `ComfyUI/models/mmaudio/nvidia/bigvgan_v2_44khz_128band_512x`

FIX
