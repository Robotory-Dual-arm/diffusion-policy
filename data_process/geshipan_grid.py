import h5py
import numpy as np
import math
import cv2

hdf5_path = "/home/baetae/diffusion-policy/common_data.hdf5"
dataset_name = "image_H"
stride = 8

with h5py.File(hdf5_path, "r") as f:
    images = f["data"]["demo_0"][dataset_name][:]  # (T,H,W,C) 가정 (RGB)

# 구간 자르기
images = images[350:]

T, H, W, C = images.shape
sampled = images[::stride]
N = len(sampled)
print("shape:", images.shape, "sampled:", N)

# ===== 그리드 크기 =====
grid_size = math.ceil(math.sqrt(N))  # 정방형에 가깝게
canvas = np.zeros((grid_size * H, grid_size * W, C), dtype=sampled.dtype)

# ===== 타일링 =====
for idx, img in enumerate(sampled):
    r = idx // grid_size
    c = idx % grid_size
    y0, y1 = r * H, (r + 1) * H
    x0, x1 = c * W, (c + 1) * W
    canvas[y0:y1, x0:x1] = img

# ===== 보기 편하게 스케일 =====
scale = 0.5
canvas_small = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

# OpenCV 표시용: RGB -> BGR
# canvas_bgr = cv2.cvtColor(canvas_small, cv2.COLOR_RGB2BGR)

cv2.imshow("grid view", canvas_small)
cv2.waitKey(0)
cv2.destroyAllWindows()
