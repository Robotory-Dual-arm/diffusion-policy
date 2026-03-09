import h5py
import numpy as np
import cv2

hdf5_path = "/home/baetae/diffusion-policy/common_data.hdf5"
dataset_name = "image_H"
stride = 25

with h5py.File(hdf5_path, "r") as f:
    images = f["data"]["demo_0"][dataset_name][:]  # (T,H,W,C), RGB 가정

T, H, W, C = images.shape
images = images[125:200]
sampled = images[::stride]
N = len(sampled)
print("shape:", images.shape, "sampled:", N)

# float로 누적 (정확한 블렌딩)
acc = np.zeros((H, W, C), dtype=np.float32)

# 각 프레임 가중치: 뒤로 갈수록 더 진하게(=최근 프레임 강조)
# 원하면 동일가중치로 바꿔도 됨.
weights = np.linspace(0.2, 0.5, N).astype(np.float32)
weights = weights / weights.sum()

for img, w in zip(sampled, weights):
    acc += img.astype(np.float32) * w

overlay = np.clip(acc, 0, 255).astype(np.uint8)  # RGB

# 보기 편하게 축소
scale = 2.0
overlay_small = cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

# OpenCV는 BGR로 표시하니까 변환해서 띄우면 색 정상
overlay_bgr = cv2.cvtColor(overlay_small, cv2.COLOR_RGB2BGR)

cv2.imshow("motion trail (overlay)", overlay_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
