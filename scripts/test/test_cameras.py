"""抓取每个 OpenCV 摄像头索引的一帧，保存为图片，方便区分哪个索引对应哪个摄像头。"""
import cv2

indices = [0, 2, 3]
for i in indices:
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"idx={i}: 无法打开")
        continue
    ret, frame = cap.read()
    if ret:
        out_path = f"C:/Users/User/Desktop/cam_idx_{i}.jpg"
        cv2.imwrite(out_path, frame)
        print(f"idx={i}: 已保存 {out_path}")
    else:
        print(f"idx={i}: 打开但读不到帧")
    cap.release()
