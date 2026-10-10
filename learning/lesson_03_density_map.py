"""
Lesson 0.3 applied: turn ShanghaiTech head points into a density map.

Each head point becomes a single "1" pixel, then the whole map is blurred
with a Gaussian filter. A Gaussian filter is sum-preserving, so the total
of the finished density map should still equal the original head count.

Requires: data/shanghaitech_sample/IMG_1.jpg and GT_IMG_1.mat
"""
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "data/shanghaitech_sample/IMG_1.jpg"
MAT_PATH = "data/shanghaitech_sample/GT_IMG_1.mat"
OUT_PATH = "data/shanghaitech_sample/IMG_1_density_map.png"

SIGMA = 15  # fixed Gaussian spread in pixels (simple version, not geometry-adaptive)

img = np.array(Image.open(IMAGE_PATH))
height, width = img.shape[:2]

mat = loadmat(MAT_PATH)
info = mat["image_info"][0, 0][0, 0]
points = info["location"]  # (N, 2), each row (x, y)
stated_count = int(info["number"][0, 0])

# start from an all-zero map, place a 1 at each head point's pixel
point_map = np.zeros((height, width), dtype=np.float32)
placed = 0
for x, y in points:
    col, row = int(round(x)), int(round(y))
    if 0 <= row < height and 0 <= col < width:
        point_map[row, col] += 1
        placed += 1

density_map = gaussian_filter(point_map, sigma=SIGMA)

print("stated count:", stated_count)
print("points placed on the map (some near edges may be clipped):", placed)
print("sum of point_map (before blur):", point_map.sum())
print("sum of density_map (after blur):", round(density_map.sum(), 2))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(img)
axes[0].scatter(points[:, 0], points[:, 1], s=8, c="red", marker="x")
axes[0].set_title(f"Head points (n={stated_count})")
axes[0].axis("off")

axes[1].imshow(img)
axes[1].imshow(density_map, cmap="jet", alpha=0.6)
axes[1].set_title(f"Density map (sum={density_map.sum():.1f})")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=120)
print(f"saved comparison figure to {OUT_PATH}")
