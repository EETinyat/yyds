import cv2
from ultralytics import YOLO
from collections import defaultdict

model = YOLO(r"E:\zhonghesheji\shuju\forPP\VOCdata2\runs\train\exp\weights\best.pt")

source = r"E:\testVIDEO\497111398-1-208.mp4"


track_history = defaultdict(lambda: [])

if __name__ == '__main__':
    captrue = cv2.VideoCapture(source)
    if not captrue.isOpened():
        print("no video")
        exit()
    fps = captrue.get(cv2.CAP_PROP_FPS)
    frame_width = captrue.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = captrue.get(cv2.CAP_PROP_FRAME_HEIGHT)

    videoWriter = None

    while True:
        success, frame = captrue.read()
        if not success:
            print("false read")
            break

        results = model.track(frame, persist=True)
        a_frame = results[0].plot()
        cv2.namedWindow("yolo track", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("yolo track", 500, 1080)
        cv2.imshow("yolo track", a_frame)
        cv2.waitKey(1)


    captrue.release()
    cv2.destroyWindow()