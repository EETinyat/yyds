from ultralytics import YOLO

model = YOLO(r"runs/train4/exp2/weights/best.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam

# Run inference on 'bus.jpg' with arguments
model.predict(r"E:\zhonghesheji\shuju\forPP\fire\JPEGImages\359.jpg", save=True,
              imgsz=640,
              conf=0.2,
              project='runs/predict',  # 项目名称（可选）
              name='exp_g2')