"""
Lesson 0.2 applied: read ShanghaiTech's .mat ground-truth annotation,
confirm it's just a list of (x, y) head points, and overlay it on the image.

Requires: data/shanghaitech_sample/IMG_1.jpg and GT_IMG_1.mat
(download with `kaggle datasets download tthien/shanghaitech -f <path>`)
"""
from scipy.io import loadmat
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "data/shanghaitech_sample/IMG_1.jpg"
MAT_PATH = "data/shanghaitech_sample/GT_IMG_1.mat"
OUT_PATH = "data/shanghaitech_sample/IMG_1_annotated.png"

mat = loadmat(MAT_PATH)

# image_info is nested as [0,0] with two fields: 'location' (Nx2 head points)
# and 'number' (the count, stored for convenience/cross-check)
info = mat["image_info"][0, 0][0, 0]
points = info["location"]  # shape (N, 2), each row is (x, y)
stated_count = int(info["number"][0, 0])

print("points array shape:", points.shape)
print("stated count (from .mat 'number' field):", stated_count)
print("actual count (len of points):", len(points))
assert len(points) == stated_count, "point count should match the stated count"

img = np.array(Image.open(IMAGE_PATH))

plt.figure(figsize=(10, 7))
plt.imshow(img)
plt.scatter(points[:, 0], points[:, 1], s=8, c="red", marker="x")
plt.title(f"IMG_1.jpg - {stated_count} head points")
plt.axis("off")
plt.savefig(OUT_PATH, bbox_inches="tight", dpi=120)
print(f"saved annotated overlay to {OUT_PATH}")
