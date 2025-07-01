import cv2
from ultralytics import YOLO


model = YOLO(r"E:\zhonghesheji\shuju\forPP\VOCdata2\runs\train\exp\weights\best.pt")

source = r"E:\testVIDEO\817377310-1-208.mp4"

if __name__ == '__main__':
    captrue = cv2.VideoCapture(source)
    if not captrue.isOpened():
        print("no video")
        exit()

    while True:
        success, frame = captrue.read()
        if not success:
            print("false read")
            break
        results = model.track(frame, persist=True)
            # 过滤置信度大于0.5的结果
        ccc = []
        if results[0].boxes.conf.numel() > 0:
            conf = results[0].boxes.conf[0]
            if conf > 0.5:
                a_frame = results
                ccc.append(a_frame)
        for res in ccc:
            a_frame = res[0].plot()
            cv2.namedWindow("yolo track", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("yolo track", a_frame)
        cv2.waitKey(1)






    captrue.release()
    cv2.destroyWindow()