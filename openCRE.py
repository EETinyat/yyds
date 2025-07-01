from ultralytics import YOLO
import cv2


# 加载自定义训练的模型
model = YOLO(r"runs/train4/exp2/weights/best.pt")

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 从摄像头捕获帧
    ret, frame = cap.read()
 
    # 如果成功捕获帧，则进行处理
    if ret:
        # 使用模型进行预测（配置为接受图像输入）
        results = model.track(frame, persist=True)
        a_frame = results[0].plot()
        cv2.imshow("yolo track", a_frame)
        cv2.waitKey(1)

        # 按下'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # 如果未能捕获帧，则打印错误并退出循环
        print("Failed to grab frame")
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()


