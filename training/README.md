# Training Module

训练模块包含分布式训练脚本和配置文件。

## 文件说明

### `trainer.py`
**分布式 DDP 训练脚本**

```bash
# 标准训练
torchrun --nproc_per_node=2 trainer.py --epochs 50

# 自定义参数
torchrun --nproc_per_node=2 trainer.py \
    --window_size 30 \
    --w_swivel 1.0 \
    --w_elbow 1.0 \
    --w_smooth 0.1
```

**参数说明**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--window_size` | 30 | 时序窗口大小 |
| `--w_swivel` | 1.0 | 拟人先验权重 |
| `--w_elbow` | 1.0 | 空间约束权重 |
| `--w_smooth` | 0.1 | 平滑约束权重 |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 512 | 批次大小 |

### `configs/`
**训练配置目录**（预留）

## 训练输出

训练完成后，模型保存在 `checkpoints/` 目录：

```
checkpoints/
├── 20260227_111614/
│   ├── best_model.pth
│   └── checkpoints/
└── ...
```

## 使用示例

```python
# 训练完整模型
torchrun --nproc_per_node=2 trainer.py --epochs 50

# 训练消融实验模型
torchrun --nproc_per_node=2 trainer.py --window_size 15 --epochs 50
torchrun --nproc_per_node=2 trainer.py --window_size 1 --epochs 50
```
