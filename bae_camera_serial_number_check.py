import pyrealsense2 as rs
import numpy as np

def list_connected_cameras():
    # Context 생성
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("No RealSense devices found.")
        return
    
    for idx, dev in enumerate(devices):
        name   = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        fw_ver = dev.get_info(rs.camera_info.firmware_version)
        print(f"[{idx}] {name}")
        print(f"    Serial Number : {serial}")
        print(f"    Firmware Ver. : {fw_ver}")

if __name__ == "__main__":
    list_connected_cameras()
