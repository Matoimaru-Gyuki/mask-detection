"""
性能对比测试 — CPU vs GPU vs GPU+TensorRT
运行: cd src && python performance_test.py
"""
import cv2
import time
import numpy as np
import os
import sys


def test_inference(engine, frames, num_warmup=5):
    """测试推理引擎性能"""
    print(f"  预热 {num_warmup} 帧...")
    for i in range(min(num_warmup, len(frames))):
        engine.infer(frames[i])

    print(f"  测试 {len(frames)} 帧...")
    total_time = 0.0
    for frame in frames:
        _, _, _, dt = engine.infer(frame)
        total_time += dt

    avg_ms = total_time / len(frames)
    fps = 1000 / avg_ms if avg_ms > 0 else 0
    return avg_ms, fps


def main():
    # 模型路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_pt = os.path.join(base_dir, "runs", "detect", "mask_detection", "exp1", "weights", "best.pt")
    model_onnx = os.path.join(base_dir, "runs", "detect", "mask_detection", "exp1", "weights", "best.onnx")
    model_engine = os.path.join(base_dir, "runs", "detect", "mask_detection", "exp1", "weights", "best.engine")

    # 检查模型文件
    if not os.path.exists(model_pt):
        print(f"[错误] 找不到 PyTorch 模型: {model_pt}")
        print("请先运行 train.py 训练模型")
        sys.exit(1)

    # 准备测试帧（使用摄像头或随机数据）
    print("准备测试帧...")
    cap = cv2.VideoCapture(0)
    frames = []
    for _ in range(50):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()

    if len(frames) < 10:
        print("摄像头不可用，使用随机数据模拟测试...")
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(50)]

    print(f"测试帧数: {len(frames)}")

    results = {}

    # --- CPU 测试 ---
    print("\n" + "=" * 50)
    print("  [1/3] CPU 推理 (ONNX Runtime)")
    print("=" * 50)
    try:
        from inference import CPUInference
        cpu = CPUInference(model_onnx if os.path.exists(model_onnx) else model_pt)
        ms, fps = test_inference(cpu, frames)
        results["CPU"] = {"avg_ms": ms, "fps": fps}
        print(f"  平均延迟: {ms:.2f} ms | FPS: {fps:.2f}")
    except Exception as e:
        print(f"  CPU 测试失败: {e}")
        # 如果没有 ONNX，用 PyTorch CPU 模式
        if os.path.exists(model_pt):
            from inference import GPUInference
            import torch
            cpu = GPUInference(model_pt)
            ms, fps = test_inference(cpu, frames)
            results["CPU"] = {"avg_ms": ms, "fps": fps}
            print(f"  平均延迟: {ms:.2f} ms | FPS: {fps:.2f} (PyTorch fallback)")

    # --- GPU 测试 ---
    print("\n" + "=" * 50)
    print("  [2/3] GPU 推理 (PyTorch CUDA)")
    print("=" * 50)
    try:
        from inference import GPUInference
        gpu = GPUInference(model_pt)
        ms, fps = test_inference(gpu, frames)
        results["GPU (PyTorch)"] = {"avg_ms": ms, "fps": fps}
        print(f"  平均延迟: {ms:.2f} ms | FPS: {fps:.2f}")
    except Exception as e:
        print(f"  GPU 测试失败: {e}")

    # --- TensorRT 测试 ---
    print("\n" + "=" * 50)
    print("  [3/3] GPU + TensorRT 推理")
    print("=" * 50)
    try:
        from inference import TensorRTInference
        trt = TensorRTInference(model_engine if os.path.exists(model_engine) else model_pt)
        ms, fps = test_inference(trt, frames)
        results["GPU + TensorRT"] = {"avg_ms": ms, "fps": fps}
        print(f"  平均延迟: {ms:.2f} ms | FPS: {fps:.2f}")
    except Exception as e:
        print(f"  TensorRT 测试失败: {e}")
        print("  请确保已安装: pip install tensorrt-cu12 onnxruntime-gpu")

    # --- 输出对比表格 ---
    print("\n" + "=" * 60)
    print("               性能对比汇总")
    print("=" * 60)
    print(f"{'模式':<20} {'平均延迟(ms)':<15} {'FPS':<10} {'加速比':<10}")
    print("-" * 60)

    baseline_ms = None
    for mode, data in results.items():
        if baseline_ms is None:
            baseline_ms = data["avg_ms"]
        speedup = baseline_ms / data["avg_ms"] if data["avg_ms"] > 0 else 0
        print(f"{mode:<20} {data['avg_ms']:<15.2f} {data['fps']:<10.2f} {speedup:<10.2f}x")

    # 保存结果
    csv_path = os.path.join(base_dir, "..", "performance_results.csv")
    try:
        import pandas as pd
        df = pd.DataFrame([
            {"Mode": k, "Avg_Latency_ms": v["avg_ms"], "FPS": v["fps"]}
            for k, v in results.items()
        ])
        df.to_csv(csv_path, index=False)
        print(f"\n结果已保存到: {csv_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()