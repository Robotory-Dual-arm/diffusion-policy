import pyrealsense2 as rs

# 연결된 장치 목록 가져오기
ctx = rs.context()
devices = ctx.query_devices()

print(f"총 연결된 RealSense 장치 수: {len(devices)}")

for i, dev in enumerate(devices):
    name = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)
    print(f"\n[{i}] 장치 이름: {name}")
    print(f"[{i}] 시리얼 번호: {serial}")

    try:
        # 각각의 장치에 대해 개별 파이프라인 구성
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # 파이프라인 시작
        profile = pipeline.start(config)

        # 컬러 센서 가져오기
        color_sensor = profile.get_device().first_color_sensor()
        print(f"[{i}] 컬러 센서 찾음: {color_sensor.get_info(rs.camera_info.name)}")

        # 설정 테스트 (예: 자동노출 꺼보기)
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        print(f"[{i}] 컬러 센서에 옵션 설정 성공")

        # 파이프라인 멈추기
        pipeline.stop()

    except Exception as e:
        print(f"[{i}] 에러 발생: {e}")
