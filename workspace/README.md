# Workspace Module

工作空间模块包含数据集分析、VR 遥操作映射和对比可视化功能。

## 文件说明

### `analyze.py` - 工作空间分析
分析训练数据集的末端位姿分布。

```bash
python analyze.py
```

**输出**：
- `dataset_workspace_limits.json` - 边界统计数据
- `dataset_workspace_3d.png` - 3D 散点图
- `dataset_workspace_2d_projections.png` - 2D 投影图
- `dataset_workspace_histograms.png` - 分布直方图
- `dataset_workspace_comprehensive.png` - 综合分析图

### `retargeter.py` - VR 坐标映射器
将 VR 控制器位置映射到数据集舒适区。

```python
from retargeter import WorkspaceRetargeter

retargeter = WorkspaceRetargeter(
    vr_json_path='vr_workspace_limits.json',
    dataset_json_path='dataset_workspace_limits.json',
    uniform_scale=True
)
T_mapped = retargeter.map_pose(T_vr)
```

### `compare.py` - 工作空间对比
对比 VR 和数据集的工作空间分布。

```bash
python compare.py \
    --vr-npy /path/to/vr_workspace_raw.npy \
    --dataset-npz /path/to/dataset.npz
```

## VR 遥操作集成

详见：[workspace_retargeting_README.md](../../workspace_retargeting_README.md)

## 使用示例

```python
from retargeter import WorkspaceRetargeter

# 初始化映射器
retargeter = WorkspaceRetargeter(
    vr_json_path='data/vr_workspace_limits.json',
    dataset_json_path='data/dataset_workspace_limits.json'
)

# 映射 VR 位姿
T_mapped = retargeter.map_pose(T_vr)
```
