import h5py
import numpy as np
from PIL import Image

h5_path = "251210/force_diffusion_data.hdf5"
image_name = "image0"

out_path = f"{image_name}.png"

with h5py.File(h5_path, "r") as f:
    img = f["data"]["demo_0"]["obs"][image_name][0]  # <-- 첫 프레임
    img = np.array(img)  # h5py dataset -> numpy

# (H,W,C) / (C,H,W) 둘 다 대응
if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC

# float이면 0~1 or 0~255 케이스 대응해서 uint8로
if np.issubdtype(img.dtype, np.floating):
    mx = float(np.nanmax(img))
    if mx <= 1.0:
        img = (img * 255.0).clip(0, 255)
    img = img.astype(np.uint8)
else:
    if img.dtype != np.uint8:
        img = img.clip(0, 255).astype(np.uint8)

# grayscale이면 (H,W)로
if img.ndim == 3 and img.shape[2] == 1:
    img = img[:, :, 0]

# img = img[:, :, ::-1]  # BGR -> RGB


Image.fromarray(img).save(out_path)
print("saved:", out_path, "shape:", img.shape, "dtype:", img.dtype)