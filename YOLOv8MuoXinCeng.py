import torch
from ultralytics import YOLO

# 加载YOLOv8n模型
model = YOLO('yolov8n.pt').model  # 获取PyTorch模型对象

# 遍历所有层并统计
conv_count = 0
c2f_conv_count = 0
sppf_conv_count = 0
detect_conv_count = 0

for name, module in model.named_modules():
    print(name)
    if isinstance(module, torch.nn.Conv2d):
        conv_count += 1
        if 'c2f' in name:
            c2f_conv_count += 1
        elif 'sppf' in name:
            sppf_conv_count += 1
        elif 'detect' in name:
            detect_conv_count += 1

print(f"总卷积层数: {conv_count}")
print(f"C2f模块内卷积层数: {c2f_conv_count}")
print(f"SPPF模块内卷积层数: {sppf_conv_count}")
print(f"检测头卷积层数: {detect_conv_count}")