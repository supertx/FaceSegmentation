"""
@author supermantx
@date 2024/8/27 17:43
"""


class Config:
    dataset_root = "/home/jiangda/tx/data/CelebAMask-HQ_112x112"

    img_root = "CelebA-HQ-img"
    mask_root = "CelebAMask-HQ-mask-anno"

    feas = ['skin', 'neck', 'hat', 'eye_g', 'hair', 'ear_r', 'neck_l', 'cloth', 'l_eye', 'r_eye', 'l_brow',
            'r_brow', 'nose', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip']

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
