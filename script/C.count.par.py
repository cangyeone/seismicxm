# benchmark_models_mac.py
import time
import argparse
import math

import torch
from models.EQLarge9l import EQLargeCNN as Model1, Loss  # noqa: F401
from models.EQLarge9a import EQLargeCNN as Model2, Loss  # noqa: F401
from models.EQLarge9b import EQLargeCNN as Model3, Loss  # noqa: F401
from models.EQLarge9o import EQLargeCNN as Model4, Loss  # noqa: F401
from models.EQLarge9 import EQLargeCNN as Model5, Loss  # noqa: F401

def human_mb(param_count: int, dtype_bytes: int) -> float:
    return param_count * dtype_bytes / 1024 / 1024


def count_params(m: torch.nn.Module):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def timed_forward(model, x, iters: int, device: torch.device, sync: bool):
    # 预热
    for _ in range(min(5, max(1, iters // 10))):
        _ = model(x)
        if sync:
            torch.mps.synchronize()

    # 正式计时
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
        if sync:
            torch.mps.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters  # 单次前向的平均秒数


def try_run_two_layouts(model, shape_btc, dtype, device, iters, sync):
    B, T, C = shape_btc
    # Layout 1: [B, T, C]
    x_btc = torch.randn(B, T, C, dtype=dtype, device=device)
    try:
        t = timed_forward(model, x_btc, iters, device, sync)
        return "BTC", t
    except Exception:
        # Layout 2: [B, C, T]
        x_bct = torch.randn(B, C, T, dtype=dtype, device=device)
        t = timed_forward(model, x_bct, iters, device, sync)
        return "BCT", t


def main():
    parser = argparse.ArgumentParser(description="Mac (CPU/MPS) 模型尺寸与推理速度基准")
    parser.add_argument("--batch", type=int, default=10, help="批量 B")
    parser.add_argument("--length", type=int, default=10240, help="时间长度 T")
    parser.add_argument("--channels", type=int, default=3, help="通道数 C")
    parser.add_argument("--iters", type=int, default=10, help="计时迭代次数")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16"], help="计算精度")
    parser.add_argument("--threads", type=int, default=None, help="CPU 线程数（仅 CPU 有效）")
    args = parser.parse_args()

    # 设备选择：优先 MPS
    use_mps = torch.backends.mps.is_available()
    device = torch.device("mps") if use_mps else torch.device("cpu")
    device = torch.device("cpu")
    sync = use_mps  # 只有 MPS 需要同步

    if args.threads and not use_mps:
        torch.set_num_threads(args.threads)

    # dtype 选择
    if args.dtype == "fp16":
        dtype = torch.float16
        dtype_bytes = 2
    else:
        dtype = torch.float32
        dtype_bytes = 4

    models = {
        "EQLarge9l": Model1,
        "EQLarge9a": Model2,
        "EQLarge9b": Model3,
        "EQLarge9o": Model4,
        "EQLarge": Model5, 
    }

    print(f"Device: {'MPS' if use_mps else 'CPU'} | DType: {args.dtype} | "
          f"B={args.batch}, T={args.length}, C={args.channels}, iters={args.iters}")
    print("-" * 90)
    header = f"{'Model':<12}{'Params(M)':>12}{'Size(MB)':>12}{'Layout':>10}{'Latency(ms)':>14}{'Throughput(sps)':>18}"
    print(header)
    print("-" * 90)

    for name, cls in models.items():
        m = cls()
        # 转精度
        try:
            m = m.to(dtype)
        except Exception:
            # 个别层不支持 FP16 时，回退 FP32
            if args.dtype == "fp16":
                print(f"[WARN] {name} 不完全支持 fp16，回退 fp32 测试。")
                dtype = torch.float32
                dtype_bytes = 4
                m = m.to(dtype)

        m = m.to(device)
        m.eval()

        total_params, _ = count_params(m)
        size_mb = human_mb(total_params, dtype_bytes)

        # 两种常见输入布局尝试
        try:
            layout, avg_secs = try_run_two_layouts(
                m, (args.batch, args.length, args.channels), dtype, device, args.iters, sync
            )
        except Exception as e:
            print(f"{name:<12}{'ERR':>12}{'ERR':>12}{'ERR':>10}{'ERR':>14}{'ERR':>18}")
            print(f"  Error: {e}")
            continue

        latency_ms = avg_secs * 1000.0
        # 每秒处理的样本数（按一次前向处理 B 个样本）
        throughput = args.batch / avg_secs

        print(f"{name:<12}{total_params/1e6:>12.2f}{size_mb:>12.2f}{layout:>10}"
              f"{latency_ms:>14.2f}{throughput:>18.2f}")

    print("-" * 90)
    print("说明：")
    print("  • Params(M) 为参数量（百万）；Size(MB) 为按当前 dtype 估算的参数内存占用。")
    print("  • Latency(ms) 为单次前向的平均耗时；Throughput(sps) 为每秒样本数（样本=波形段）。")
    print("  • 自动尝试 [B,T,C] 与 [B,C,T] 两种输入布局。若两者皆不符你的模型，请手动改造输入。")
    print("  • 在 Mac 上建议优先使用 MPS + fp16（若模型支持）以获得更高吞吐。")


if __name__ == "__main__":
    main()
