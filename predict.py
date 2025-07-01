
from ultralytics import YOLO

model = YOLO(r"runs/train4/exp2/weights/best.pt")

model.predict(r"E:\xuexi\shengduxuexi\yolotrainVOC\images\train\fire_dp222_1.jpg", save=True,
              imgsz=640,
              conf=0.2,  # 置信度
              project='runs/predict',  # 项目名称（可选）
              name='exp_g')