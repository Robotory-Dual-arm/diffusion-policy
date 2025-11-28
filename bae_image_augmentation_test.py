import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import math

origin_image_path = "bae_image_augmentation_test/origin_image.png"
augmented_image_path = "bae_image_augmentation_test"

origin_image = Image.open(origin_image_path).convert("RGB")

# print('int', np.array(origin_image), np.array(origin_image).shape)
# float_image = T.ToTensor()(origin_image)
# print('float', float_image, float_image.shape)

# breakpoint()

transform = T.Compose([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
                       T.RandomGrayscale(p=0.005)])
                       

num_augment = 14

# augmented 이미지 plt 시각화
max_cols = 5
total = num_augment + 1
cols = min(total, max_cols)
rows = math.ceil(total / max_cols)

fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
axes = np.array(axes).reshape(-1)

axes[0].imshow(origin_image)
axes[0].set_title("Original Image")
axes[0].axis("off")

for i in range(num_augment):
    augmented_image = transform(origin_image)
    axes[i+1].imshow(augmented_image)
    axes[i+1].set_title(f"Augmented Image {i+1}")
    axes[i+1].axis("off")

for j in range(total, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

# augmented 이미지 저장
# for i in range(num_augment):
#     augmented_image = transform(origin_image)
#     augmented_image_pil = T.ToPILImage()(augmented_image)
#     augmented_image_pil.save(f"{augmented_image_path}/augmented_image_{i}.png")
