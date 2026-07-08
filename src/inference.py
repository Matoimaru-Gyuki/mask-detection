"""
多后端推理引擎 — 支持 CPU / GPU / TensorRT 三种模式
"""
import time
import numpy as np
import cv2
import os


class BaseInference:
    """推理基类"""

    def __init__(self, conf_thres=0.5, iou_thres=0.45):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.warmup_done = False

    def preprocess(self, image):
        """预处理: resize, normalize, BGR->RGB, HWC->CHW"""
        img = cv2.resize(image, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)    # 添加 batch 维度
        return img

    def infer(self, image):
        raise NotImplementedError

    def draw_boxes(self, image, boxes, scores, classes, class_names=None):
        """绘制检测框"""
        if class_names is None:
            class_names = {0: "R_mask", 1: "W_mask", 2: "N_mask"}

        # 颜色: 正确=绿, 错误=黄, 未戴=红
        COLORS = {0: (0, 255, 0), 1: (0, 215, 255), 2: (0, 0, 255)}

        result = image.copy()
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            cls = int(cls)
            label = class_names.get(cls, f"cls_{cls}")
            color = COLORS.get(cls, (128, 128, 128))

            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {score:.2f}"
            cv2.putText(result, text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return result


class CPUInference(BaseInference):
    """ONNX Runtime CPU 推理"""

    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, image):
        input_tensor = self.preprocess(image)
        start = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        elapsed = (time.perf_counter() - start) * 1000
        boxes, scores, classes = self._postprocess(outputs[0], image.shape[:2])
        return boxes, scores, classes, elapsed

    def _postprocess(self, output, orig_shape):
        """简化后处理: 从 YOLOv8 ONNX 输出解析检测框"""
        # output shape: (1, 84, 8400) — 84 = 4(bbox) + 80(COCO classes)
        output = np.squeeze(output[0]) if isinstance(output, list) else np.squeeze(output)
        if output.ndim == 3:
            output = output[0]

        boxes, scores, classes = [], [], []
        h, w = orig_shape

        for i in range(output.shape[1]):
            row = output[:, i]
            cls_id = np.argmax(row[4:])
            score = row[4 + cls_id]
            if score < self.conf_thres:
                continue

            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            x1 = int((cx - bw / 2) * w / 640)
            y1 = int((cy - bh / 2) * h / 640)
            x2 = int((cx + bw / 2) * w / 640)
            y2 = int((cy + bh / 2) * h / 640)

            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
            classes.append(int(cls_id))

        # NMS
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)
            if len(indices) > 0:
                indices = indices.flatten()
                boxes = [boxes[i] for i in indices]
                scores = [scores[i] for i in indices]
                classes = [classes[i] for i in indices]

        return boxes, scores, classes


class GPUInference(BaseInference):
    """PyTorch / Ultralytics GPU 推理"""

    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        from ultralytics import YOLO
        import torch
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def infer(self, image):
        start = time.perf_counter()
        results = self.model(image, conf=self.conf_thres, iou=self.iou_thres,
                             device=self.device, verbose=False)
        elapsed = (time.perf_counter() - start) * 1000

        boxes, scores, classes = [], [], []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    boxes.append(box.xyxy[0].cpu().numpy())
                    scores.append(float(box.conf[0].cpu().numpy()))
                    classes.append(int(box.cls[0].cpu().numpy()))

        return boxes, scores, classes, elapsed


class TensorRTInference(BaseInference):
    """
    TensorRT 推理 — 直接加载 .engine 文件，使用 TensorRT Python API
    用 torch 管理 GPU 内存，无需 pycuda
    """

    def __init__(self, engine_path, **kwargs):
        super().__init__(**kwargs)
        import tensorrt as trt
        import torch

        self.trt = trt
        self.torch = torch
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.stream = torch.cuda.current_stream().cuda_stream

        # 加载 engine 文件
        print("TensorRT 引擎加载中...")
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # 获取输入输出信息
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        # 用 torch 分配 GPU 内存（FP32, 连续内存）
        self.d_input = torch.empty(self.input_shape, dtype=torch.float32, device='cuda')
        self.d_output = torch.empty(self.output_shape, dtype=torch.float32, device='cuda')

        # 预热
        print("TensorRT 引擎预热中...")
        dummy = np.random.randn(*self.input_shape).astype(np.float32)
        for _ in range(3):
            self._infer_raw(dummy)
        self.warmup_done = True
        print("TensorRT 预热完成")

    def _infer_raw(self, input_tensor):
        """底层推理：输入 numpy (1,3,640,640) -> 输出 numpy"""
        # CPU -> GPU
        self.d_input.copy_(self.torch.from_numpy(np.ascontiguousarray(input_tensor)))

        self.context.set_tensor_address(self.input_name, self.d_input.data_ptr())
        self.context.set_tensor_address(self.output_name, self.d_output.data_ptr())

        self.context.execute_async_v3(self.stream)
        self.torch.cuda.synchronize()

        return self.d_output.cpu().numpy()

    def infer(self, image):
        input_tensor = self.preprocess(image)
        start = time.perf_counter()
        output = self._infer_raw(input_tensor)
        elapsed = (time.perf_counter() - start) * 1000

        boxes, scores, classes = [], [], []
        h, w = image.shape[:2]

        # YOLOv8 TensorRT 输出格式: (1, 84, 8400)
        preds = np.squeeze(output)
        preds = preds.T  # (8400, 84)

        for pred in preds:
            cls_id = np.argmax(pred[4:])
            score = pred[4 + cls_id]
            if score < self.conf_thres:
                continue

            cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
            x1 = int((cx - bw / 2) * w / 640)
            y1 = int((cy - bh / 2) * h / 640)
            x2 = int((cx + bw / 2) * w / 640)
            y2 = int((cy + bh / 2) * h / 640)

            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
            classes.append(int(cls_id))

        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)
            if len(indices) > 0:
                indices = indices.flatten()
                boxes = [boxes[i] for i in indices]
                scores = [scores[i] for i in indices]
                classes = [classes[i] for i in indices]

        return boxes, scores, classes, elapsed


def create_inference(mode, model_path):
    """工厂函数: 根据模式创建推理引擎"""
    mode = mode.lower()
    if mode == "cpu":
        print(f"[CPU] 加载 ONNX 模型: {model_path}")
        return CPUInference(model_path)
    elif mode == "gpu":
        print(f"[GPU] 加载 PyTorch 模型: {model_path}")
        return GPUInference(model_path)
    elif mode == "tensorrt":
        print(f"[TensorRT] 加载 Engine 文件: {model_path}")
        return TensorRTInference(model_path)
    else:
        raise ValueError(f"未知模式: {mode}，可选: cpu / gpu / tensorrt")