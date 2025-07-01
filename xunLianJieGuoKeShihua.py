import pandas as pd

# 读取用户上传的文件
file_path = r'runs/train2/exp_v3/results.csv'
data = pd.read_csv(file_path)

# 查看数据的基本信息和前几行，确保文件格式正确
data_info = data.info()
data_head = data.head()

data_info, data_head

import matplotlib.pyplot as plt

# 提取关键指标
epochs = data['epoch']
train_box_loss = data['train/box_loss']
train_cls_loss = data['train/cls_loss']
train_dfl_loss = data['train/dfl_loss']
val_box_loss = data['val/box_loss']
val_cls_loss = data['val/cls_loss']
val_dfl_loss = data['val/dfl_loss']
precision = data['metrics/precision(B)']
recall = data['metrics/recall(B)']
mAP50 = data['metrics/mAP50(B)']
mAP50_95 = data['metrics/mAP50-95(B)']

# 设置图形大小
plt.figure(figsize=(18, 12))

# 绘制训练和验证损失
plt.subplot(2, 2, 1)
plt.plot(epochs, train_box_loss, label='Train Box Loss', color='blue')
plt.plot(epochs, val_box_loss, label='Validation Box Loss', color='red')
plt.title('Box Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(epochs, train_cls_loss, label='Train Classification Loss', color='blue')
plt.plot(epochs, val_cls_loss, label='Validation Classification Loss', color='red')
plt.title('Classification Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# 绘制性能指标
plt.subplot(2, 2, 3)
plt.plot(epochs, precision, label='Precision', color='green')
plt.plot(epochs, recall, label='Recall', color='orange')
plt.plot(epochs, mAP50, label='mAP50', color='purple')
plt.plot(epochs, mAP50_95, label='mAP50-95', color='brown')
plt.title('Performance Metrics Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.legend()
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(epochs, train_dfl_loss, label='Train DFL Loss', color='blue')
plt.plot(epochs, val_dfl_loss, label='Validation DFL Loss', color='red')
plt.title('DFL Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

