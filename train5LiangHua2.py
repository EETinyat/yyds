import torch
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType


def train_optimized_model():
    # ✅ 关键修改1：使用YOLO配置对象设置EMA
    model = YOLO('yolov8n.pt')  # 从配置文件创建模型


    # 优化后的训练参数
    train_args = {
        'data': 'data.yaml',
        'epochs': 200,
        'imgsz': 640,
        'batch': 32,
        'optimizer': 'Adam',
        'lr0': 0.001,
        'lrf': 0.01,
        'cos_lr': True,
        'amp': True,
        'project': 'runs/train2',
        'name': 'exp_v3',
        'device': 0,
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
        'cls': 0.5,  # 分类损失权重
    }

    # 硬件加速配置
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    results = model.train(**train_args)
    return model


def export_and_quantize(model):
    # ONNX导出配置
    export_args = {
        'format': 'onnx',
        'imgsz': 640,
        'opset': 17,
        'simplify': True,
        'nms': True,
        'half': True,
        'name': 'runs/train2/exp_v3/weights/best.onnx'
    }
    model.export(**export_args)

    # 动态量化
    quantize_dynamic(
        model_input='runs/train2/exp_v3/weights/best.onnx',
        model_output='best_quant_v3.onnx',
        weight_type=QuantType.QUInt8,
        optimize_model=True,
        op_types_to_quantize=['Conv', 'MatMul', 'Add']
    )


if __name__ == '__main__':
    try:
        # 阶段1：训练优化模型
        model = train_optimized_model()

        # 阶段2：导出与量化
        export_and_quantize(model)

        print("""
        ✅ 流程完成！验证步骤：
        1. 精度验证: yolo val model=best_quant_v3.onnx
        2. 速度测试: yolo benchmark model=best_quant_v3.onnx
        3. 比较模型大小: du -h best_quant_v3.onnx
        """)

    except Exception as e:
        print(f"❌ 错误发生: {str(e)}")
        print("""
        🛠️ 排查建议：
        1. 确认ultralytics版本 >= 8.0.200 (pip install ultralytics --upgrade)
        2. 检查data.yaml中的路径是否为绝对路径
        3. 尝试使用CPU模式训练: device='cpu'
        4. 减少batch_size到16测试
        """)