"""
YOLOv8 口罩检测模型训练脚本
运行: cd src && python train.py
"""
import torch
from ultralytics import YOLO
import os
import sys


def main():
    # 检查 CUDA
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"训练设备: {'GPU (' + torch.cuda.get_device_name(0) + ')' if device == 0 else 'CPU'}")

    # 数据集配置路径
    data_yaml = os.path.join(os.path.dirname(__file__), "..", "configs", "mask_dataset.yaml")
    if not os.path.exists(data_yaml):
        print(f"[错误] 数据集配置文件不存在: {data_yaml}")
        print("请先准备数据集并创建 configs/mask_dataset.yaml")
        sys.exit(1)

    # 加载预训练模型
    model = YOLO("yolov8n.pt")  # n=nano(轻量), 可选 s=small, m=medium

    # 训练
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        workers=2,
        project="mask_detection",
        name="exp1",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )

    print("\n训练完成！")
    best_pt = os.path.join("runs", "detect", "mask_detection", "exp1", "weights", "best.pt")
    print(f"最佳模型: {best_pt}")

    # 导出 ONNX
    print("\n导出 ONNX...")
    model = YOLO(best_pt)
    model.export(format="onnx", imgsz=640, dynamic=True)
    print("ONNX 导出完成")

    # 尝试导出 TensorRT
    try:
        print("\n导出 TensorRT 引擎 (FP16)...")
        model.export(format="engine", imgsz=640, half=True)
        print("TensorRT 引擎导出完成")
    except Exception as e:
        print(f"TensorRT 导出失败: {e}")
        print("可稍后手动导出: model.export(format='engine', imgsz=640, half=True)")


if __name__ == "__main__":
    main()