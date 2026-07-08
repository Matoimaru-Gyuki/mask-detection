"""
训练损失曲线可视化
运行: cd src && python visualize_loss.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_loss_curves(csv_path, save_path="loss_curves.png"):
    """读取训练结果 CSV 并绘制损失和指标曲线"""
    if not os.path.exists(csv_path):
        print(f"[错误] 找不到结果文件: {csv_path}")
        print("请先运行训练脚本 train.py")
        return

    df = pd.read_csv(csv_path)
    # 清除列名空格
    df.columns = df.columns.str.strip()

    # 取有效 epoch
    df = df.dropna(subset=['epoch'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('YOLOv8 口罩检测训练结果', fontsize=16, fontweight='bold')

    # --- 训练损失 ---
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['train/box_loss'], label='Box Loss', color='#e74c3c')
    ax.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss', color='#3498db')
    ax.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', color='#2ecc71')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 验证损失 ---
    ax = axes[0, 1]
    ax.plot(df['epoch'], df['val/box_loss'], label='Box Loss', color='#e74c3c')
    ax.plot(df['epoch'], df['val/cls_loss'], label='Cls Loss', color='#3498db')
    ax.plot(df['epoch'], df['val/dfl_loss'], label='DFL Loss', color='#2ecc71')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- mAP 指标 ---
    ax = axes[1, 0]
    ax.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50', color='#9b59b6', linewidth=2)
    ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95', color='#f39c12', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title('Detection Metrics (mAP)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Precision / Recall ---
    ax = axes[1, 1]
    ax.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='#1abc9c', linewidth=2)
    ax.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='#e67e22', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Precision & Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"损失曲线已保存到: {save_path}")


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "mask_detection", "exp1", "results.csv")
    save_path = os.path.join(os.path.dirname(__file__), "..", "loss_curves.png")
    plot_loss_curves(csv_path, save_path)


if __name__ == "__main__":
    main()