# Examples

示例代码包含 VR 遥操作接口和数据下载脚本。

## 文件说明

### `vr_teleoperation.py`
**VR 遥操作示例代码**

演示如何使用 PiM-IK 进行 VR 遥操作控制。

```python
from vr_teleoperation import main

main(
    use_hand_tracking=False,
    binocular=False,
    img_shape=(720, 1280),
    display_mode='pass-through'
)
```

### `download_hf_dataset.py`
**HuggingFace 数据集下载工具**

```bash
python download_hf_dataset.py --dataset ACCAD_CMU
```

## 使用示例

```python
# VR 遥操作
python vr_teleoperation.py --input-mode vr

# 下载数据
python download_hf_dataset.py --dataset GRAB
```
