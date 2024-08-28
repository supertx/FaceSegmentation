"""
@author supermantx
@date 2024/8/27 17:04
将人脸图片和mask映射成112x112的像素
"""
import os

import cv2 as cv

# [] count the number of files in the raw and new directories
# raw_dir = "/home/jiangda/tx/data/CelebAMask-HQ/CelebAMask-HQ-mask-anno/"
# new_dir = "/home/jiangda/tx/data/CelebAMask-HQ_112x112/CelebAMask-HQ-mask-anno/"
# count_raw = 0
# count_new = 0
# for _, _, filenames in os.walk(raw_dir):
#     count_raw += len(filenames)
# for _, _, filenames in os.walk(new_dir):
#     count_new += len(filenames)
# print(count_raw, count_new)

# []
# for dirpath, dirnames, filenames in os.walk(raw_dir):
#     if len(dirnames) > 0:
#         continue
#     for filename in filenames:
#         if filename.split(".")[-1] != "png":
#             continue
#         index = filename.split("_")[0]
#         if not os.path.exists(os.path.join(new_dir, index)):
#             os.makedirs(os.path.join(new_dir, index))
#         img = cv.imread(os.path.join(dirpath, filename))
#         img = cv.resize(img, (112, 112))
#         cv.imwrite(os.path.join(new_dir, index, filename), img)

# []
raw_dir = "/home/jiangda/tx/data/CelebAMask-HQ/CelebA-HQ-img/"
new_dir = "/home/jiangda/tx/data/CelebAMask-HQ_112x112/CelebA-HQ-img/"
os.makedirs(new_dir, exist_ok=True)
for dirpath, dirnames, filenames in os.walk(raw_dir):
    if len(dirnames) > 0:
        continue
    for filename in filenames:
        if filename.split(".")[-1] != "jpg":
            continue
        img = cv.imread(os.path.join(dirpath, filename))
        img = cv.resize(img, (112, 112))
        cv.imwrite(os.path.join(new_dir, filename), img)