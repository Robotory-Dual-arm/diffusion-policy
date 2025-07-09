import pyrealsense2 as rs

# 파이프라인 설정
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("❌ No RealSense devices found.")
else:
    for dev in devices:
        print(f"🔎 Found device: {dev.get_info(rs.camera_info.name)}")
        sensors = dev.query_sensors()
        for i, sensor in enumerate(sensors):
            print(f"  [{i}] Sensor name: {sensor.get_info(rs.camera_info.name)}")
            print(f"     Type: {type(sensor)}")