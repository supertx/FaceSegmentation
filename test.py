"""
@author supermantx
@date 2024/8/27 11:36
"""
import os

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


colors_rgb = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "brown": (165, 42, 42),
    "pink": (255, 192, 203),
    "lime": (0, 255, 0),
    "navy": (0, 0, 128),
    "olive": (128, 128, 0),
    "maroon": (128, 0, 0),
    "gray": (128, 128, 128),
    "silver": (192, 192, 192),
    "gold": (255, 215, 0),
    "sky": (135, 206, 235),
    "beige": (245, 245, 220),
    "ivory": (255, 255, 240),
    "tan": (210, 180, 140),
    "lavender": (230, 230, 250),
    "salmon": (250, 128, 114),
    "coral": (255, 127, 80),
    "khaki": (240, 230, 140),
    "indigo": (75, 0, 130),
}

# index = ["skin", "hair", "eye_g", "r_ear", "l_ear", "l_eye", "r_eye",
#          "nose", "r_brow", "l_brow", "mouth", "u_lip", "l_lip", "neck", "ear_r", "neck_l", "cloth"]
index = ['skin', 'neck', 'hat', 'eye_g', 'hair', 'ear_r', 'neck_l', 'cloth', 'l_eye', 'r_eye', 'l_brow',
         'r_brow', 'nose', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip']

# generate mask with the order of face feature
mask = np.zeros((512, 512, 3))
for file, color_rgb in zip(index, colors_rgb.values()):
    if not os.path.isfile(os.path.join("test", "29999_" + file + ".png")):
        continue
    imread = cv.imread(os.path.join("test", "29999_" + file + ".png"))
    imread = imread / 255
    mask *= (1.0 - imread)
    imread = imread * color_rgb
    mask += imread

raw_img = cv.imread("29999.jpg")
raw_img = cv.cvtColor(raw_img, cv.COLOR_BGR2RGB)
raw_img = cv.resize(raw_img, (112, 112))
subplot = plt.subplot(1, 3, 1)
subplot.imshow(raw_img)
subplot.set_title("raw_img")
subplot = plt.subplot(1, 3, 2)
mask = mask.astype(np.uint8)
mask = cv.resize(mask, (112, 112))
subplot.imshow(mask)
subplot.set_title("mask")
subplot = plt.subplot(1, 3, 3)
subplot.imshow((raw_img * 0.5 + mask * 0.5) / 255)
subplot.set_title("result")
plt.show()

