import torch
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType, preprocess
import onnx

DEVICE = 0
BATCH_SIZE = 16
AMP_MODE = True
NUM_WORKERS = 4

def train_model():
    model = YOLO('yolov8n.pt')

    # 已验证的有效参数组合
    train_args = {
        'data': 'data.yaml',
        'epochs': 200,
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
        'name': 'exp4',
        'seed': 17,
        'nms': True
    }

    torch.backends.cudnn.benchmark = True  # 启用CuDNN自动优化器
    torch.set_float32_matmul_precision('high')  # 加速矩阵运算

    results = model.train(**train_args)
    return model
if __name__ == '__main__':
    model = train_model()

    # ✅ 关键修改1：导出ONNX时指定opset=17并禁用动态维度
    export_path = 'runs/train2/exp4/weights/best.onnx'
    model.export(
        format='onnx',
        imgsz=640,
        opset=17,  # 强制使用兼容性更好的opset版本
        dynamic=False,  # 固定输入输出维度
        simplify=True,  # 移除冗余节点
        nms=True,  # 包含NMS层
        name=export_path  # 指定完整导出路径
    )

    quantize_dynamic(
        model_input=export_path,  # 使用字符串路径而非模型对象
        model_output='best_quant.onnx',
        weight_type=QuantType.QUInt8,
        optimize_model=True
    )

    # ✅ 极简模型验证
    onnx.checker.check_model(onnx.load('best_quant.onnx'))
    print("量化验证通过！")