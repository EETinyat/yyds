from ultralytics import YOLO
import torch
import torch_pruning as tp

DEVICE = 0
BATCH_SIZE = 16  # 直接设置有效batch_size
AMP_MODE = True
NUM_WORKERS = 4  # 必须小于等于CPU物理核心数


def get_pruning_layers(model):
    pruning_layers = []
    ignored_layers = []

    # 打印所有层名用于调试
    print("=" * 50 + " 模型层结构分析 " + "=" * 50)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"[检测到卷积层] {name}")
            # 排除所有检测头相关层（根据实际打印调整）
            if any(key in name for key in ['detect', 'cv2', 'cv3', 'dfl']):
                print(f"└── 已排除：{name}")
                ignored_layers.append(module)
            else:
                pruning_layers.append(module)
    print("=" * 100)
    return pruning_layers, ignored_layers

# 1. 先正常训练模型
def train_model():
    model = YOLO('yolov8n.pt')

    # 已验证的有效参数组合
    train_args = {
        'data': 'data.yaml',
        'epochs': 3,
        'imgsz': 640,
        'device': DEVICE,
        'workers': NUM_WORKERS,
        'batch': BATCH_SIZE,  # 直接设置总batch大小
        'optimizer': 'Adam',  # YOLOv8对AdamW支持不稳定
        'lr0': 0.01,
        'cos_lr': True,
        'amp': AMP_MODE,
        'box': 7.5,
        'cls': 0.5,
        'project': 'runs/train2',
        'name': 'exp3_optimized',
        'seed': 17
    }

    # 显存优化配置（关键！）
    torch.backends.cudnn.benchmark = True  # 启用CuDNN自动优化器
    torch.set_float32_matmul_precision('high')  # 加速矩阵运算

    results = model.train(**train_args)
    return model


# 2. 剪枝函数
def prune_yolov8n(trained_model, prune_ratio=0.3):
    model = trained_model.model.cpu()

    # 获取剪枝层并验证排除情况
    pruning_layers, ignored_layers = get_pruning_layers(model)

    # 剪枝配置（关键参数验证）
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=torch.randn(1, 3, 640, 640),
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=prune_ratio,
        ignored_layers=ignored_layers,
        round_to=8,
        global_pruning=False  # 改为局部剪枝确保安全
    )
    pruner.step()

    # 保存完整模型结构（关键修改！）
    # 保存剪枝后的模型到文件
    pruned_model_path = "pruned_yolov8n.pt"
    model.save(pruned_model_path)  # 显式保存模型

    # 从保存的文件重新加载模型
    pruned_model = YOLO(pruned_model_path)  # 使用文件路径加载

    return pruned_model


def fine_tune(pruned_model_path):
    # 直接加载完整剪枝模型
    model = YOLO(pruned_model_path)

    # 微调参数（保持与原始训练相同输入尺寸）
    fine_tune_args = {
        'data': 'data.yaml',
        'epochs': 5,
        'imgsz': 640,  # 必须与剪枝时输入尺寸一致
        'device': DEVICE,
        'workers': NUM_WORKERS,
        'batch': BATCH_SIZE,
        'lr0': 1e-4,
        'cos_lr': True,
        'amp': AMP_MODE,
        'project': 'runs/train2',
        'name': 'exp3_pruned',
        'seed': 17
    }

    # 显存优化配置
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    model.train(**fine_tune_args)
    model.export(format='onnx')


if __name__ == '__main__':
    # 两步执行流程（避免内存泄漏）
    trained = train_model()
    pruned = prune_yolov8n(trained)
    fine_tune(pruned)