#!/usr/bin/env python3
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import threading

# ---------- 단일 카메라용 클래스 ----------
class Pipeline:
    def __init__(self, serial):
        self.serial = serial
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # start pipeline
        self.pipeline.start(self.config)

        # 워밍업/재시도 루프 (안정화용)
        
        self.pipeline.wait_for_frames(timeout_ms=1000)
            

        print(f"[INFO] Realsense camera {serial} initialized.")

    def read(self, timeout_ms=1000):
        frames = self.pipeline.wait_for_frames(timeout_ms)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError(f"No color frame from {self.serial}")
        return np.asanyarray(color.get_data())

    def stop(self):
        try:
            self.pipeline.stop()
        except:
            pass


# ---------- 두 대 카메라 띄우기 ----------
def show_camera(cam: Pipeline, win_name: str):
    while True:
        try:
            img = cam.read()
        except Exception as e:
            img = np.zeros((480,640,3), dtype=np.uint8)
            cv2.putText(img, str(e), (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow(win_name, img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC로 종료
            break
    cam.stop()


if __name__ == "__main__":
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) < 2:
        print("카메라 두 대가 필요합니다.")
        exit(1)

    s0 = devices[0].get_info(rs.camera_info.serial_number)
    s1 = devices[1].get_info(rs.camera_info.serial_number)

    cam0 = Pipeline(s0)
    cam1 = Pipeline(s1)

    # 스레드로 동시에 띄우기
    t0 = threading.Thread(target=show_camera, args=(cam0,"cam0"), daemon=True)
    t1 = threading.Thread(target=show_camera, args=(cam1,"cam1"), daemon=True)
    t0.start(); t1.start()

    t0.join(); t1.join()
    cv2.destroyAllWindows()
