#!/usr/bin/env python3
# Minimal dual RealSense RGB viewer (640x480 @ 30fps).
# - Auto-picks the first two connected RealSense devices (no CLI flags).
# - Shows two windows: cam0 (left), cam1 (right).
# - RGB only. Press ESC to quit.

import sys
import time
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except Exception as e:
    print("Failed to import pyrealsense2. Install librealsense + pyrealsense2.", file=sys.stderr)
    raise

def main():
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) < 2:
        print("Need at least TWO RealSense devices connected.")
        for d in devs:
            print(" -", d.get_info(rs.camera_info.serial_number),
                  d.get_info(rs.camera_info.name))
        sys.exit(1)

    serial0 = devs[0].get_info(rs.camera_info.serial_number)
    serial1 = devs[1].get_info(rs.camera_info.serial_number)
    print("Using devices:", serial0, "and", serial1)

    # Create pipelines and configs for RGB streams
    pipe0, cfg0 = rs.pipeline(), rs.config()
    cfg0.enable_device(serial0)
    cfg0.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipe1, cfg1 = rs.pipeline(), rs.config()
    cfg1.enable_device(serial1)
    cfg1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start
    prof0 = pipe0.start(cfg0)
    prof1 = pipe1.start(cfg1)

    # Small warmup
    for _ in range(10):
        pipe0.poll_for_frames()
        pipe1.poll_for_frames()
        time.sleep(0.01)

    cv2.namedWindow("cam0", cv2.WINDOW_NORMAL)
    cv2.namedWindow("cam1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("cam0", 640, 480)
    cv2.resizeWindow("cam1", 640, 480)

    print("Press ESC to exit.")
    try:
        while True:
            f0 = pipe0.wait_for_frames(1000)
            f1 = pipe1.wait_for_frames(1000)

            c0 = f0.get_color_frame()
            c1 = f1.get_color_frame()

            if not c0 or not c1:
                continue

            img0 = np.asanyarray(c0.get_data())
            img1 = np.asanyarray(c1.get_data())

            cv2.imshow("cam0", img0)
            cv2.imshow("cam1", img1)

            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        try: pipe0.stop()
        except Exception: pass
        try: pipe1.stop()
        except Exception: pass
        cv2.destroyAllWindows()
        time.sleep(0.2)

if __name__ == "__main__":
    main()