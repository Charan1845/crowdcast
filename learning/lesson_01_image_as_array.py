"""
Lesson 0.1 applied: confirm image-as-array basics on one real ShanghaiTech image.

Requires: data/shanghaitech_sample/IMG_1.jpg (download with scripts/download_sample.sh
or `kaggle datasets download tthien/shanghaitech -f ShanghaiTech/part_A/test_data/images/IMG_1.jpg`)
"""
from PIL import Image
import numpy as np

IMAGE_PATH = "data/shanghaitech_sample/IMG_1.jpg"

img = Image.open(IMAGE_PATH)
arr = np.array(img)

print("PIL size (width, height):", img.size)
print("PIL mode:", img.mode)
print()
print("NumPy shape (height, width, channels):", arr.shape)
print("dtype:", arr.dtype)
print("min pixel value:", arr.min())
print("max pixel value:", arr.max())
print("mean brightness:", round(arr.mean(), 2))
print()
print("one pixel, row=0 col=0 (R,G,B):", arr[0, 0])
