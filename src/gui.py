"""
口罩佩戴检测系统 GUI — 支持图片/视频/摄像头检测
运行: cd src && python gui.py
"""
import sys
import os
import cv2
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QMessageBox,
    QGroupBox, QGridLayout
)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont


class InferenceThread(QThread):
    """推理线程，避免阻塞 UI"""
    result_ready = pyqtSignal(object, float, object, object, object)

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.frame = None
        self.running = False

    def set_frame(self, frame):
        self.frame = frame

    def run(self):
        while self.running:
            if self.frame is not None:
                frame = self.frame.copy()
                boxes, scores, classes, elapsed = self.engine.infer(frame)
                annotated = self.engine.draw_boxes(frame, boxes, scores, classes)
                self.result_ready.emit(annotated, elapsed, boxes, scores, classes)
            self.msleep(1)


class MaskDetectGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("口罩佩戴检测系统")
        self.setGeometry(100, 100, 1100, 700)

        # 状态
        self.engine = None
        self.cap = None
        self.mode = "gpu"
        self.is_running = False
        self.current_frame = None
        self.fps_history = []

        # 模型路径
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_pt = os.path.join(self.base_dir, "runs", "detect", "mask_detection", "exp1", "weights", "best.pt")
        self.model_onnx = os.path.join(self.base_dir, "runs", "detect", "mask_detection", "exp1", "weights", "best.onnx")
        self.model_engine = os.path.join(self.base_dir, "runs", "detect", "mask_detection", "exp1", "weights", "best.engine")

        self.init_ui()
        self.init_engine()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)

        # --- 控制栏 ---
        control_group = QGroupBox("控制面板")
        control_layout = QGridLayout(control_group)

        # 模式选择
        control_layout.addWidget(QLabel("推理后端:"), 0, 0)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["GPU (PyTorch)", "CPU (ONNX)", "GPU + TensorRT"])
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)
        control_layout.addWidget(self.combo_mode, 0, 1)

        # 检测按钮
        self.btn_image = QPushButton("图片检测")
        self.btn_image.clicked.connect(self.on_image)
        control_layout.addWidget(self.btn_image, 0, 2)

        self.btn_video = QPushButton("视频检测")
        self.btn_video.clicked.connect(self.on_video)
        control_layout.addWidget(self.btn_video, 0, 3)

        self.btn_camera = QPushButton("摄像头检测")
        self.btn_camera.clicked.connect(self.on_camera)
        control_layout.addWidget(self.btn_camera, 0, 4)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setStyleSheet("background-color: #c0392b; color: white;")
        control_layout.addWidget(self.btn_stop, 0, 5)

        main_layout.addWidget(control_group)

        # --- 显示区 ---
        display_group = QGroupBox("检测画面")
        display_layout = QVBoxLayout(display_group)

        self.label_display = QLabel()
        self.label_display.setAlignment(Qt.AlignCenter)
        self.label_display.setMinimumSize(800, 450)
        self.label_display.setStyleSheet("background-color: #1a1a2e; border-radius: 8px;")
        self.label_display.setText("请选择检测模式：图片 / 视频 / 摄像头")
        display_layout.addWidget(self.label_display)

        main_layout.addWidget(display_group)

        # --- 状态栏 ---
        status_layout = QHBoxLayout()
        self.label_status = QLabel("状态: 就绪")
        self.label_fps = QLabel("FPS: --")
        self.label_count = QLabel("检测: R=0, W=0, N=0")
        status_layout.addWidget(self.label_status)
        status_layout.addStretch()
        status_layout.addWidget(self.label_fps)
        status_layout.addWidget(self.label_count)
        main_layout.addLayout(status_layout)

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def init_engine(self):
        """初始化推理引擎"""
        try:
            from inference import create_inference

            mode_map = {
                "GPU (PyTorch)": "gpu",
                "CPU (ONNX)": "cpu",
                "GPU + TensorRT": "tensorrt",
            }
            self.mode = mode_map[self.combo_mode.currentText()]

            if self.mode == "tensorrt":
                model_path = self.model_engine if os.path.exists(self.model_engine) else self.model_pt
            else:
                model_path = self.model_pt

            if not os.path.exists(model_path):
                self.label_status.setText("状态: 未找到模型文件，请先训练")
                return

            self.engine = create_inference(self.mode, model_path)
            self.label_status.setText(f"状态: 引擎就绪 ({self.combo_mode.currentText()})")

        except Exception as e:
            self.label_status.setText(f"状态: 引擎加载失败 - {str(e)[:50]}")

    def on_mode_changed(self):
        self.stop()
        self.init_engine()

    def on_image(self):
        self.stop()
        fname, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not fname:
            return

        if self.engine is None:
            self.init_engine()
            if self.engine is None:
                QMessageBox.warning(self, "错误", "推理引擎未就绪，请检查模型文件")
                return

        img = cv2.imread(fname)
        if img is None:
            QMessageBox.warning(self, "错误", "无法读取图片")
            return

        boxes, scores, classes, elapsed = self.engine.infer(img)
        annotated = self.engine.draw_boxes(img, boxes, scores, classes)

        self.show_image(annotated)
        mask_count = sum(1 for c in classes if c == 0)
        wrong_count = sum(1 for c in classes if c == 1)
        none_count = sum(1 for c in classes if c == 2)
        self.label_status.setText(f"状态: 图片检测完成 ({elapsed:.1f}ms)")
        self.label_fps.setText(f"延迟: {elapsed:.1f}ms")
        self.label_count.setText(f"检测: R={mask_count}, W={wrong_count}, N={none_count}")

    def on_video(self):
        self.stop()
        fname, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if not fname:
            return

        if self.engine is None:
            self.init_engine()
            if self.engine is None:
                return

        self.cap = cv2.VideoCapture(fname)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", "无法打开视频文件")
            return

        self.is_running = True
        self.timer.start(30)
        self.label_status.setText(f"状态: 视频检测中 ({self.combo_mode.currentText()})")

    def on_camera(self):
        self.stop()

        if self.engine is None:
            self.init_engine()
            if self.engine is None:
                return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", "无法打开摄像头")
            return

        self.is_running = True
        self.timer.start(30)
        self.label_status.setText(f"状态: 摄像头检测中 ({self.combo_mode.currentText()})")

    def update_frame(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            self.label_status.setText("状态: 视频流结束")
            return

        if self.engine is None:
            return

        try:
            boxes, scores, classes, elapsed = self.engine.infer(frame)
            annotated = self.engine.draw_boxes(frame, boxes, scores, classes)

            # 添加 FPS 信息
            fps = 1000 / elapsed if elapsed > 0 else 0
            cv2.putText(annotated, f"FPS: {fps:.1f} | {elapsed:.1f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 添加统计
            mask_count = sum(1 for c in classes if c == 0)
            wrong_count = sum(1 for c in classes if c == 1)
            none_count = sum(1 for c in classes if c == 2)
            cv2.putText(annotated, f"R_mask: {mask_count} | W_mask: {wrong_count} | N_mask: {none_count}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            self.show_image(annotated)
            self.label_fps.setText(f"FPS: {fps:.1f}")
            self.label_count.setText(f"检测: R={mask_count}, W={wrong_count}, N={none_count}")

        except Exception as e:
            self.label_status.setText(f"状态: 推理错误 - {str(e)[:50]}")

    def show_image(self, img):
        """将 OpenCV 图像显示到 QLabel"""
        h, w, ch = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled = qt_img.scaled(
            self.label_display.width(), self.label_display.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label_display.setPixmap(QPixmap.fromImage(scaled))

    def stop(self):
        self.timer.stop()
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.label_display.clear()
        self.label_display.setText("检测已停止")
        self.label_status.setText("状态: 已停止")
        self.label_fps.setText("FPS: --")
        self.label_count.setText("检测: R=0, W=0, N=0")

    def closeEvent(self, event):
        self.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))

    # 样式
    app.setStyleSheet("""
        QMainWindow { background-color: #2c2c3e; }
        QGroupBox {
            font-weight: bold; border: 1px solid #555; border-radius: 8px;
            margin-top: 12px; padding-top: 16px; color: #ddd;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
        QPushButton {
            padding: 8px 16px; border-radius: 6px; border: 1px solid #555;
            background-color: #3a3a5c; color: #eee; font-weight: bold;
        }
        QPushButton:hover { background-color: #4a4a7c; }
        QPushButton:pressed { background-color: #2a2a4c; }
        QComboBox {
            padding: 6px 12px; border-radius: 6px; border: 1px solid #555;
            background-color: #3a3a5c; color: #eee;
        }
        QLabel { color: #ccc; }
    """)

    win = MaskDetectGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()