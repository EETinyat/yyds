import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置seaborn主题（替代原plt.style.use）
sns.set_theme(style="darkgrid", font_scale=1.1)  # 优化字体和样式

# 加载数据
df_normal = pd.read_csv('runs/train3/exp/results.csv')
df_pruned = pd.read_csv('runs/train2/exp3_pruned/results.csv')
df_quantized = pd.read_csv('runs/train2/exp_v3/results.csv')
df_pruned_quantized = pd.read_csv('runs/train4/exp2/results.csv')

# 定义指标
metrics = [
    ('metrics/mAP50(B)', 'mAP50(B)'),
    ('metrics/mAP50-95(B)', 'mAP50-95(B)'),
    ('metrics/precision(B)', 'Precision(B)'),
    ('metrics/recall(B)', 'Recall(B)'),
    ('train/box_loss', 'Training Box Loss'),
    ('val/box_loss', 'Validation Box Loss')
]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
axes = axes.flatten()

for idx, (col, title) in enumerate(metrics):
    ax = axes[idx]

    # 剪枝模型曲线（调整标记参数）
    ax.plot(
        df_pruned['epoch'],
        df_pruned[col],
        label='Pruned',
        color='#1f77b4',
        linewidth=1.5,  # 减小线宽
        marker='o',  # 圆形标记
        markersize=4,  # 缩小标记尺寸
        markevery=5,  # 每5个epoch显示一个标记
        alpha=0.8  # 设置透明度
    )

    # 量化模型曲线（调整标记参数）
    ax.plot(
        df_quantized['epoch'],
        df_quantized[col],
        label='Quantized',
        color='#ff7f0e',
        linewidth=1.5,
        marker='s',  # 正方形标记
        markersize=4,
        markevery=5,
        alpha=0.8
    )

    ax.plot(
        df_pruned_quantized['epoch'],
        df_pruned_quantized[col],
        label='Pruned_Quantized',
        color='#844200',
        linewidth=1.5,  # 减小线宽
        marker='x',  #
        markersize=4,  # 缩小标记尺寸
        markevery=5,  # 每5个epoch显示一个标记
        alpha=0.8  # 设置透明度
    )

    ax.plot(
        df_normal['epoch'],
        df_normal[col],
        label='Normal',
        color='#006030',
        linewidth=1.5,  # 减小线宽
        marker='p',  #
        markersize=4,  # 缩小标记尺寸
        markevery=5,  # 每5个epoch显示一个标记
        alpha=0.8  # 设置透明度
    )

    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)
    ax.set_title(f'{title} Comparison')
    ax.legend()

# 隐藏多余子图
for j in range(len(metrics), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig('optimized_trend.png', dpi=300, bbox_inches='tight')
plt.show()