"""
PiM-IK 骨干网络延迟与显存基准测试

用于消融实验：对比 Mamba / LSTM / Causal Transformer 在单帧流式推理场景下的
- 计算延迟 (Latency)
- 峰值显存占用 (Peak VRAM)
- 模型参数量 (Parameters)

使用场景：机器人实时控制，要求低延迟 (<10ms for 1kHz control)

作者: PiM-IK 项目
日期: 2025-02-28
"""

import torch
import time
from pim_ik_net import PiM_IK_Net


def format_number(num: float, unit: str = "") -> str:
    """格式化数字，保留合适的小数位数"""
    if num >= 1000:
        return f"{num/1000:.1f}K{unit}"
    elif num >= 1:
        return f"{num:.2f}{unit}"
    else:
        return f"{num*1000:.1f}m{unit}"


def benchmark_model(model: torch.nn.Module, T_ee: torch.Tensor,
                    num_warmup: int = 100, num_iters: int = 1000) -> dict:
    """
    基准测试单个模型

    Args:
        model: 待测试模型
        T_ee: 输入张量
        num_warmup: 预热次数
        num_iters: 测试迭代次数

    Returns:
        dict: 包含 latency_ms, vram_mb, params 等指标
    """
    model.eval()
    device = T_ee.device

    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters())

    # 预热：让 GPU 频率稳定
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(T_ee)
        torch.cuda.synchronize()

    # 显存测试
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.no_grad():
        _ = model(T_ee)
    torch.cuda.synchronize()

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # 延迟测试
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(T_ee)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time_s = end_time - start_time
    avg_latency_ms = (total_time_s / num_iters) * 1000

    return {
        'latency_ms': avg_latency_ms,
        'vram_mb': peak_vram_mb,
        'params': num_params
    }


def print_markdown_table(results: dict):
    """打印精美的 Markdown 表格"""
    print("\n" + "=" * 80)
    print("PiM-IK 骨干网络性能基准测试结果")
    print("=" * 80)
    print("\n测试配置:")
    print("  - 输入形状: (B=1, W=30, 4, 4)")
    print("  - 隐空间维度: d_model=256")
    print("  - 堆叠层数: num_layers=4")
    print("  - 预热次数: 100")
    print("  - 测试迭代: 1000")
    print("  - 设备: CUDA")

    print("\n" + "| Model | Params | Latency | Peak VRAM |")
    print("|-------|--------|---------|-----------|")

    for model_name, metrics in results.items():
        params_k = metrics['params'] / 1000
        latency = metrics['latency_ms']
        vram = metrics['vram_mb']

        # 参数量列
        params_str = f"{params_k:.0f}K"

        # 延迟列（根据是否适合实时控制着色）
        if latency < 10:
            latency_str = f"**{latency:.2f}** ✅"  # 适合 1kHz 控制
        elif latency < 33:
            latency_str = f"{latency:.2f} ⚠️"  # 适合 30Hz 控制
        else:
            latency_str = f"{latency:.2f} ❌"  # 延迟过高

        # 显存列
        vram_str = f"{vram:.1f}"

        print(f"| {model_name} | {params_str} | {latency_str} | {vram_str} |")

    print("\n说明:")
    print("  - ✅ : 延迟 < 10ms，适合 1kHz 实时控制")
    print("  - ⚠️  : 延迟 < 33ms，适合 30Hz 视觉伺服")
    print("  - ❌ : 延迟过高，不适合实时控制")
    print("=" * 80 + "\n")


def main():
    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("错误: 需要 CUDA 设备运行基准测试")
        return

    device = torch.device('cuda:0')

    # 测试配置（与训练脚本保持一致）
    d_model = 256
    num_layers = 4
    batch_size = 1
    window_size = 30

    # 构造输入张量（模拟真实推理环境）
    T_ee = torch.randn(batch_size, window_size, 4, 4, device=device)

    print(f"\n初始化模型...")
    print(f"  设备: {device}")
    print(f"  输入形状: {T_ee.shape}")

    # 初始化三个模型
    models = {
        'Mamba': PiM_IK_Net(d_model=d_model, num_layers=num_layers, backbone_type='mamba'),
        'LSTM': PiM_IK_Net(d_model=d_model, num_layers=num_layers, backbone_type='lstm'),
        'Transformer': PiM_IK_Net(d_model=d_model, num_layers=num_layers, backbone_type='transformer'),
    }

    # 将模型移到 GPU
    for name, model in models.items():
        models[name] = model.to(device)
        print(f"  ✅ {name} 已加载到 GPU")

    # 运行基准测试
    results = {}
    for name, model in models.items():
        print(f"\n正在测试 {name}...")
        results[name] = benchmark_model(model, T_ee)
        print(f"  延迟: {results[name]['latency_ms']:.3f} ms")
        print(f"  显存: {results[name]['vram_mb']:.1f} MB")
        print(f"  参数: {results[name]['params']:,}")

    # 打印结果表格
    print_markdown_table(results)

    # 额外分析
    print("📊 性能对比分析:")

    # 找出最快的模型
    fastest = min(results.items(), key=lambda x: x[1]['latency_ms'])
    print(f"  • 最快: {fastest[0]} ({fastest[1]['latency_ms']:.2f} ms)")

    # 计算相对 Mamba 的加速比
    if 'Mamba' in results:
        mamba_latency = results['Mamba']['latency_ms']
        for name, metrics in results.items():
            if name != 'Mamba':
                ratio = metrics['latency_ms'] / mamba_latency
                if ratio > 1:
                    print(f"  • Mamba 相比 {name} 快 {ratio:.2f}x")
                else:
                    print(f"  • {name} 相比 Mamba 快 {1/ratio:.2f}x")

    # 实时控制适用性
    print(f"\n🤖 实时控制适用性 (1kHz 控制 < 10ms):")
    for name, metrics in results.items():
        status = "✅ 适合" if metrics['latency_ms'] < 10 else "❌ 不适合"
        print(f"  • {name}: {metrics['latency_ms']:.2f} ms -> {status}")


if __name__ == "__main__":
    main()
