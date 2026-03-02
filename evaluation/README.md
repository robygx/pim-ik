# Evaluation Module

评估模块包含模型评估脚本和性能基准测试。

## 文件说明

### `evaluate.py`
**模型评估脚本**

```bash
python evaluate.py \
    --checkpoint checkpoints/20260227_111614/best_model.pth \
    --data_path /path/to/GRAB_training_data_with_swivel.npz
```

**输出指标**：
- Swivel MAE (°)
- Elbow Position Error (mm)
- Jerk (平滑度)

### `benchmark_latency.py`
**推理延迟基准测试**

```bash
python benchmark_latency.py \
    --checkpoint checkpoints/20260227_111614/best_model.pth \
    --num_runs 1000
```

测试结果示例：
```
平均推理延迟: 0.45 ms
P50 延迟: 0.42 ms
P99 延迟: 0.58 ms
```

## 使用示例

```python
from evaluate import load_model, compute_metrics

# 加载模型
model = load_model('checkpoints/20260227_111614/best_model.pth')

# 评估
metrics = compute_metrics(model, data_path)
print(f"MAE: {metrics['mae']:.2f}°")
print(f"Jerk: {metrics['jerk']:.4f}")
```
