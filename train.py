from ultralytics import YOLO

from ultralytics import YOLO

# 加载预训练的 YOLO 模型
model = YOLO('yolov8n.pt')

# 开始训练
if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    results = model.train(data='data.yaml',
                          epochs=100,
                          imgsz=640,
                          device=[0, ],
                          workers=0,
                          batch=4,
                          cache=True,
                          optimizer='Adam',
                          lr0=0.01,
                          lrf=0.01,
                          box=7.5,  # (float) 盒损失增益
                          cls=0.5,  # (float) 类别损失增益（与像素比例）
                          )  # 开始训练




