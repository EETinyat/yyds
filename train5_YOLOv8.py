import torch
from ultralytics import YOLO
# 在训练代码最前面添加环境修复（解决 90% 的 cuDNN 错误）
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 定位具体出错位置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 减少显存碎片
torch.backends.cudnn.enabled = True       # 确保 cuDNN 已启用
torch.backends.cudnn.benchmark = False    # 输入尺寸变化时需关闭
torch.backends.cudnn.deterministic = True # 增强稳定性（略微降低速度）

def train_fast():
    # ✅ 关键加速技巧：轻量化模型 + 极致硬件利用
    model = YOLO('yolov8n.pt')  # 使用nano版本（最小模型）

    # 硬件级优化配置
    torch.backends.cudnn.benchmark = True  # 固定输入尺寸时加速30%
    torch.set_float32_matmul_precision('high')  # 矩阵加速（AMP开启时需测试稳定性）

    # ⚡ 极速训练参数（速度优先）
    train_args = {
        'data': 'data.yaml',
        'epochs': 200,
        'imgsz': 640,
        'batch': 32,  # ✅ 最大化batch size（根据显存调整）
        'optimizer': 'SGD',  # ✅ SGD比Adam快15%~20%
        'lr0': 0.01,  # SGD需要更大初始学习率
        'momentum': 0.937,  # SGD必须配合动量
        'weight_decay': 0.0005,  # 正则化防止过拟合
        'workers': 4,  # ✅ 根据CPU核心数设置（推荐CPU逻辑核心数的75%）
        'device': 0,  # 使用单GPU（多GPU设置如[0,1]提速50%+）
        'amp': True,  # 混合精度加速30%~50%
        # 'close_mosaic': 10,  # ✅ 最后10个epoch关闭Mosaic（稳定训练）
        'mosaic': 0.8,  # 降低Mosaic概率减少计算
        'mixup': 0.0,  # 关闭MixUp（提速5%~10%）
        'copy_paste': 0.0,  # 关闭复制粘贴（减少CPU负载）
        'cache': 'ram',  # ✅ 数据集缓存到内存（SSD用户用disk）
        'single_cls': True,  # 如果是单类别任务，开启可提速
        'val': False,  # ❗️ 关闭验证（仅推荐预训练时）
        'pretrained': False,  # 避免重复加载权重
        # 数据增强简化
        # 'mosaic': 0.8,  # 保持增强效果但降低概率
        # 'mixup': 0.0,  # 关闭耗时增强
        'close_mosaic': 15,  # 最后15个epoch关闭Mosaic
        'project': 'runs/train3',
        'name': 'YOLOv8n_NOM_1'
    }

    results = model.train(**train_args)
    return model


if __name__ == '__main__':
    train_fast()