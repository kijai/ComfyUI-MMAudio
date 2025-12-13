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

# 重要更新（2025-11-17）

## ✨ 智能時間插值：讓短影片也能生成完整音訊

### 核心功能
**當影片幀數不足時，自動使用時間插值（temporal interpolation）擴充影片幀數，確保音訊時長與指定的 `duration` 完全匹配。**

這意味著即使您的影片只有 3 秒，也能生成 6 秒的音訊，影片會平滑地慢動作播放來填補時間。

### 問題背景
之前版本中，當影片幀數不足以支援指定的 `duration` 時，程式會自動縮短音訊時長，導致：
- 設定 `duration=6.31` 秒，實際只生成 3-4 秒音訊
- 影片後半段沒有聲音，體驗不佳

**原因分析**：
- 101 幀 @ 16 fps = 6.31 秒影片時長
- 但 MMAudio 的 Sync 模型需要 @ 25 fps，即需要 158 幀
- 幀數不足 → 被迫縮短時長 → 音訊只有 3.84 秒

### 解決方案：智能時間插值

現在採用 **trilinear interpolation（三線性插值）** 在時間維度上擴充影片幀數：

1. **自動檢測幀數不足**：當 `total_frames < required_frames` 時觸發
2. **平滑時間插值**：使用 PyTorch 的 3D 插值功能，在時間軸上生成中間幀
3. **保持視覺流暢**：插值後的影片看起來像慢動作，沒有突兀的跳幀

### 技術細節

```python
# 當 101 幀不足以支援 6.31 秒 @ 25 fps (需要 158 幀) 時
# 使用 3D 插值擴充：
video_5d = video_frames.unsqueeze(0)  # (1, C, T, H, W)
interpolated = F.interpolate(
    video_5d,
    size=(158, H, W),  # 擴充時間維度從 101 → 158
    mode='trilinear',   # 三線性插值
    align_corners=False
)
# 結果：平滑生成 57 個中間幀
```

### 使用說明

**基本使用**（推薦）：
```
duration: 6.31          # 想要的音訊時長
input_fps: 16.0         # 影片實際幀率（ComfyUI 預設）
```

**結果**：
- ✅ 自動檢測影片只有 101 幀（6.31 秒 @ 16 fps）
- ✅ 自動插值擴充到 CLIP 需要的 50 幀 @ 8 fps
- ✅ 自動插值擴充到 SYNC 需要的 158 幀 @ 25 fps
- ✅ 生成完整的 6.31 秒音訊

**注意事項**：
- 插值會讓影片看起來像慢動作（但通常不明顯）
- 建議 `duration` 不要超過影片實際時長太多（2 倍以內效果最佳）
- 如果影片幀數充足，不會進行插值，直接使用原始幀

### 效能優化

1. **GPU 加速的時間插值**：使用 PyTorch 內建的 trilinear interpolation，充分利用 GPU
2. **批次處理**：所有幀一次性處理，避免逐幀操作
3. **記憶體高效**：使用 `contiguous()` 優化記憶體佈局

### 範例日誌輸出

**情境 1：幀數不足，需要插值**
```
輸入影片: 101 幀 @ 16.0 fps (實際時長 6.31s)
目標音訊時長: 6.31s
需要的幀數: CLIP=50 @ 8.0 fps, SYNC=158 @ 25.0 fps
SYNC: 影片幀數 101 < 需求 158，使用時間插值擴充
✅ 處理完成: clip=50 幀 (6.25s @ 8.0 fps), sync=158 幀 (6.32s @ 25.0 fps), 音訊時長=6.31s
```

**情境 2：幀數充足，不需插值**
```
輸入影片: 200 幀 @ 16.0 fps (實際時長 12.5s)
目標音訊時長: 6.0s
需要的幀數: CLIP=48 @ 8.0 fps, SYNC=150 @ 25.0 fps
✅ 處理完成: clip=48 幀 (6.0s @ 8.0 fps), sync=150 幀 (6.0s @ 25.0 fps), 音訊時長=6.0s
```

### 參數說明

- **`duration`**：期望的音訊時長（秒），現在會嚴格遵守此參數
- **`input_fps`**（可選，預設 16.0）：輸入影片的幀率，用於計算和顯示資訊
  - 16 fps：ComfyUI 預設
  - 24 fps：電影標準
  - 30 fps：常見影片格式
  - 60 fps：高幀率影片