"""
环境检测脚本 — 验证所有依赖是否正确安装
运行: python src/check_env.py
"""
import sys

def check_pytorch():
    print("=" * 50)
    print("1. PyTorch & CUDA")
    print("=" * 50)
    try:
        import torch
        print(f"   PyTorch 版本: {torch.__version__}")
        print(f"   CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")
            print(f"   GPU 数量: {torch.cuda.device_count()}")
            print(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"   PyTorch CUDA 版本: {torch.version.cuda}")
            # 简单运算测试
            x = torch.randn(3, 3).cuda()
            y = torch.mm(x, x.T)
            print(f"   GPU 张量运算测试: 通过")
        else:
            print("   [警告] CUDA 不可用！请确认安装了 PyTorch CUDA 版:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
        return torch.cuda.is_available()
    except ImportError:
        print("   [错误] PyTorch 未安装")
        return False

def check_tensorrt():
    print("\n" + "=" * 50)
    print("2. TensorRT")
    print("=" * 50)
    try:
        import tensorrt as trt
        print(f"   TensorRT 版本: {trt.__version__}")
        builder = trt.Builder(trt.Logger())
        print(f"   Builder 创建: 通过")
        return True
    except ImportError:
        print("   [警告] TensorRT 未安装，将跳过 TensorRT 加速")
        print("   安装命令: pip install tensorrt-cu12")
        return False
    except Exception as e:
        print(f"   [警告] TensorRT 加载异常: {e}")
        return False

def check_onnx():
    print("\n" + "=" * 50)
    print("3. ONNX Runtime")
    print("=" * 50)
    try:
        import onnx
        print(f"   ONNX 版本: {onnx.__version__}")
    except ImportError:
        print("   [警告] ONNX 未安装: pip install onnx")

    try:
        import onnxruntime as ort
        print(f"   ONNX Runtime 版本: {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"   可用执行提供程序: {providers}")
        has_trt = "TensorrtExecutionProvider" in providers
        has_cuda = "CUDAExecutionProvider" in providers
        if has_trt:
            print(f"   TensorRT EP: 可用")
        if has_cuda:
            print(f"   CUDA EP: 可用")
        return has_trt or has_cuda
    except ImportError:
        print("   [警告] ONNX Runtime 未安装: pip install onnxruntime-gpu")
        return False

def check_ultralytics():
    print("\n" + "=" * 50)
    print("4. Ultralytics YOLO")
    print("=" * 50)
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"   Ultralytics 版本: {ultralytics.__version__}")
        return True
    except ImportError:
        print("   [错误] Ultralytics 未安装: pip install ultralytics")
        return False

def check_opencv():
    print("\n" + "=" * 50)
    print("5. OpenCV")
    print("=" * 50)
    try:
        import cv2
        print(f"   OpenCV 版本: {cv2.__version__}")
        return True
    except ImportError:
        print("   [错误] OpenCV 未安装: pip install opencv-python")
        return False

def check_gui():
    print("\n" + "=" * 50)
    print("6. PyQt5 (GUI)")
    print("=" * 50)
    try:
        from PyQt5.QtWidgets import QApplication
        print(f"   PyQt5: 可用")
        return True
    except ImportError:
        print("   [警告] PyQt5 未安装，GUI 不可用: pip install pyqt5")
        return False

def check_others():
    print("\n" + "=" * 50)
    print("7. 其他依赖")
    print("=" * 50)
    for name, pkg in [("matplotlib", "matplotlib"), ("pandas", "pandas"), ("tqdm", "tqdm")]:
        try:
            __import__(pkg)
            print(f"   {name}: 可用")
        except ImportError:
            print(f"   [警告] {name} 未安装: pip install {pkg}")

def main():
    print("\n" + "=" * 50)
    print("     口罩检测系统 — 环境检测")
    print("=" * 50)
    print(f"   Python 版本: {sys.version}")

    results = {}
    results["pytorch"] = check_pytorch()
    results["tensorrt"] = check_tensorrt()
    results["onnx"] = check_onnx()
    results["yolo"] = check_ultralytics()
    results["opencv"] = check_opencv()
    results["gui"] = check_gui()
    check_others()

    print("\n" + "=" * 50)
    print("     检测总结")
    print("=" * 50)

    all_ok = True
    required = {"pytorch": results["pytorch"], "yolo": results["yolo"], "opencv": results["opencv"]}
    for name, ok in required.items():
        status = "通过" if ok else "失败"
        print(f"   [{status}] {name}")

    if results["tensorrt"]:
        print(f"   [通过] TensorRT — 可在 GPU + TensorRT 模式下运行")
    else:
        print(f"   [跳过] TensorRT — 仅支持 CPU / GPU 模式")

    if all(required.values()):
        print("\n核心环境就绪，可以开始训练和推理！")
    else:
        print("\n请先安装缺失的依赖，参考 requirements.txt")

if __name__ == "__main__":
    main()