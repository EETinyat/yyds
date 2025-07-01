from ultralytics import YOLO
import onnxruntime as ort

# ✅ 验证ONNX模型可用性
onnx_path = 'runs/train2/exp3_pruned/weights/best.onnx'
print(ort.get_available_providers())  # 应显示['CUDAExecutionProvider', 'CPUExecutionProvider']

# ✅ 加载模型时指定输入尺寸
model = YOLO(onnx_path, task='detect')

# ✅ 执行预测
results = model.predict(
    r"E:\zhonghesheji\shuju\forPP\fire\JPEGImages\359.jpg",
    imgsz=640,
    conf=0.2,
    device='cpu',  # ✅ 暂时使用CPU测试
    nms=False,  # ✅ 禁用内置NMS（如果导出时已禁用）
    project='runs/predict',  # 项目名称（可选）
    name='exp_2'
)