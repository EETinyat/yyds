import torch
from ultralytics import YOLO
def get_pruning_layers(model):
    # 需要剪枝的目标层
    pruning_layers = []
    # 需要排除的敏感层
    ignored_layers = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # 排除检测头层
            if 'detect' in name:
                ignored_layers.append(module)
            # 排除最后1个C2f的输出层
            elif 'backbone.c2f.3' in name:  # 根据实际层名调整
                ignored_layers.append(module)
            else:
                pruning_layers.append(module)

    return pruning_layers, ignored_layers


# 加载YOLOv8n模型
model = YOLO('yolov8n.pt').model  # 获取PyTorch模型对象
pruning_layers, ignored_layers = get_pruning_layers(model)
# print(f"可剪枝层数: {len(pruning_layers)}, 排除层数: {len(ignored_layers)}")

import torch_pruning as tp


def prune_yolov8n(prune_ratio=0.5):
    # 加载模型
    model = YOLO('yolov8n.pt').model

    # 获取剪枝层
    pruning_layers, ignored_layers = get_pruning_layers(model)

    # 定义剪枝策略
    importance = tp.importance.MagnitudeImportance(p=1)  # L1范数

    # 创建剪枝器
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=torch.randn(1, 3, 640, 640),
        importance=importance,
        pruning_ratio=prune_ratio,
        ignored_layers=ignored_layers,
        global_pruning=True
    )

    # 执行剪枝
    pruner.step()

    # 验证剪枝效果
    # print("===== 剪枝后模型结构 =====")
    # print(model)

    return model


# 执行剪枝（剪枝50%通道）
pruned_model = prune_yolov8n(prune_ratio=0.5)