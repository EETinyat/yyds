import torch
import torch.nn as nn
from ultralytics import YOLO
import torch_pruning as tp
import torch.ao.quantization as quant

DEVICE = 0  # 量化通常在CPU上执行
BATCH_SIZE = 16
AMP_MODE = False  # 量化时需要禁用AMP
NUM_WORKERS = 4


def get_pruning_layers(model):
    pruning_layers = []
    ignored_layers = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if any(key in name for key in ['detect', 'cv2', 'cv3', 'dfl']):
                ignored_layers.append(module)
            else:
                pruning_layers.append(module)
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
        'project': 'runs/train4',
        'name': 'exp',
        'seed': 17,
        'box': 7.5,
        'cls': 0.5
    }
    torch.backends.cudnn.benchmark = True
    results = model.train(**train_args)
    return model


def prune_yolov8n(trained_model, prune_ratio=0.3):
    original_yolo = trained_model
    model = original_yolo.model.cpu()

    pruning_layers, ignored_layers = get_pruning_layers(model)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=torch.randn(1, 3, 640, 640),
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=prune_ratio,
        ignored_layers=ignored_layers,
        round_to=8,
        global_pruning=False
    )

    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        original_output = model(dummy_input)
    pruner.step()

    pruned_model_path = "pruned_yolov8n.pt"
    original_yolo.save(pruned_model_path)
    return pruned_model_path


def apply_quantization(model_path):
    # 加载微调后的模型
    model = YOLO(model_path).model.cpu()

    # 动态量化
    quantized_model = quant.quantize_dynamic(
        model,
        {nn.Conv2d, nn.Linear},
        dtype=torch.qint8
    )

    # 保存量化模型
    quantized_path = "quantized_yolov8n.pt"
    torch.save(quantized_model.state_dict(), quantized_path)

    # 导出为ONNX
    dummy_input = torch.randn(1, 3, 640, 640)
    torch.onnx.export(quantized_model,
                      dummy_input,
                      "quantized_yolov8n.onnx",
                      opset_version=13,
                      input_names=['images'],
                      output_names=['output'],
                      dynamic_axes={'images': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})


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
        'project': 'runs/train4',
        'name': 'exp2',
        'seed': 17,
        'box': 7.5,
        'cls': 0.5
    }
    model.train(**fine_tune_args)
    apply_quantization(pruned_model_path)  # 应用量化


if __name__ == '__main__':
    trained = train_model()
    pruned_path = prune_yolov8n(trained)
    fine_tune(pruned_path)