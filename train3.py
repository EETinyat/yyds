import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torchvision==0.20 is incompatible with torch==2.4.*")
from ultralytics import YOLO

# 加载预训练的 YOLO 模型
model = YOLO('yolov5mu.pt')

# 开始训练
if __name__ == '__main__':
    model = YOLO('yolov5mu.pt')
    results = model.train(data='data.yaml',
                          epochs=200,
                          imgsz=640,
                          device=[0, ],
                          workers=4,
                          batch=32,
                          cache='ram',
                          optimizer='Adam',
                          lr0=0.01,
                          lrf=0.001,
                          dropout=0.1,
                          weight_decay=0.0005,
                          box=7.5,  # (float) 盒损失增益
                          cls=0.5,  # (float) 类别损失增益（与像素比例）
                          project='runs/train5_m',
                          name='exp',
                          verbose=True,
                          seed=17,
                          cos_lr=True,
                          amp=True
                          )  # 开始训练
