import os
from ultralytics import YOLO
import torch
import torch_pruning as tp

DEVICE = 0
BATCH_SIZE = 32
AMP_MODE = True
NUM_WORKERS = 4

def get_pruning_layers(model):
    pruning_layers = []
    ignored_layers = []

    # print("=" * 50 + " 模型层结构分析 " + "=" * 50)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # print(f"[检测到卷积层] {name}")
            if any(key in name for key in ['detect', 'cv2', 'cv3', 'dfl']):
                # print(f"└── 已排除：{name}")
                ignored_layers.append(module)
            else:
                pruning_layers.append(module)
    print("=" * 100)
    return pruning_layers, ignored_layers


def train_model():
    model = YOLO('yolov8n.pt')
    train_args = {
        'data': 'data.yaml',
        'epochs': 200,
        'imgsz': 640,
        'device': DEVICE,
        'workers': NUM_WORKERS,
        'batch': BATCH_SIZE,
        'optimizer': 'Adam',
        'lr0': 0.001,
        'lrf': 0.01,
        'cos_lr': True,
        'amp': AMP_MODE,
        'project': 'runs/train2',
        'name': 'exp3_optimized',
        'seed': 17,
        'hsv_h': 0.015,  # 色调增强
        'hsv_s': 0.7,  # 饱和度增强
        'hsv_v': 0.4,  # 明度增强
        'translate': 0.2,  # 平移增强
        'scale': 0.9,  # 缩放增强
        'fliplr': 0.5,  # 水平翻转概率
        'mosaic': 1.0,  # Mosaic数据增强
        'mixup': 0.2,  # MixUp增强比例
        'copy_paste': 0.5,  # 复制粘贴增强
        'box': 7.5,  # 调整边界框损失权重
        'cls': 0.5  # 分类损失权重
    }
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    results = model.train(**train_args)
    return model


def prune_yolov8n(trained_model, prune_ratio=0.3):
    original_yolo = trained_model
    model = original_yolo.model.cpu()

    pruning_layers, ignored_layers = get_pruning_layers(model)

    # 创建剪枝器（确保example_inputs与训练尺寸一致）
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=torch.randn(1, 3, 640, 640),
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=prune_ratio,
        ignored_layers=ignored_layers,
        round_to=8,
        global_pruning=False
    )
    # 剪枝前验证模型可运行性
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        original_output = model(dummy_input)
    print("原始模型前向验证通过")

    pruner.step()

    # 关键步骤：重建YOLO对象并保存
    pruned_model_path = "pruned_yolov8n.pt"
    original_yolo.save(pruned_model_path)
    return pruned_model_path  # 必须返回新加载的YOLO对象


def fine_tune(pruned_model_path):
    model = YOLO(pruned_model_path)
    fine_tune_args = {
        'data': 'data.yaml',
        'epochs': 200,
        'imgsz': 640,
        'device': DEVICE,
        'workers': NUM_WORKERS,
        'batch': BATCH_SIZE,
        'optimizer': 'Adam',
        'lr0': 0.001,
        'lrf': 0.01,
        'cos_lr': True,
        'amp': AMP_MODE,
        'project': 'runs/train2',
        'name': 'exp3_optimized',
        'seed': 17,
        'hsv_h': 0.015,  # 色调增强
        'hsv_s': 0.7,  # 饱和度增强
        'hsv_v': 0.4,  # 明度增强
        'translate': 0.2,  # 平移增强
        'scale': 0.9,  # 缩放增强
        'fliplr': 0.5,  # 水平翻转概率
        'mosaic': 1.0,  # Mosaic数据增强
        'mixup': 0.2,  # MixUp增强比例
        'copy_paste': 0.5,  # 复制粘贴增强
        'box': 7.5,  # 调整边界框损失权重
        'cls': 0.5  # 分类损失权重
    }
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    model.train(**fine_tune_args)
    model.export(format='onnx')


if __name__ == '__main__':
    trained = train_model()
    pruned_path = prune_yolov8n(trained)  # 获取保存路径
    fine_tune(pruned_path)  # 传递路径而非对象
    # 在主函数添加验证
    # print("剪枝模型路径类型:", type(pruned_path))  # 应该显示 <class 'str'>
    # print("文件存在:", os.path.exists(pruned_path))