"""
读取素材的类
"""

from typing import Tuple

import cv2
import numpy as np

import tools


class Stuff:
    """
    素材
    """

    def __init__(self, name: str, pt1: Tuple[int, int], pt2: Tuple[int, int] = None, wh: Tuple[int, int] = None, rotate: bool = False):
        self.name = name
        self.x1 = pt1[0]
        self.y1 = pt1[1]
        if pt2:
            self.x2 = pt2[0]
            self.y2 = pt2[1]

            self.w = self.x2 - self.x1
            self.h = self.y2 - self.y1
        if wh:
            self.w = wh[0]
            self.h = wh[1]
            self.x2 = self.x1 + self.w
            self.y2 = self.y1 + self.h
        self.rotate = rotate

        self.bright_img = None
        self.bright_dark_img = None
        self.dark_img = None

    def get_img(self, raw_img: np.ndarray):
        img = raw_img[self.y1:self.y2, self.x1:self.x2]
        if self.rotate:
            return np.transpose(img, (1, 0, 2))[::-1]
        return img

    def get_bright_img(self, raw_rgb: np.ndarray, block_img: np.ndarray, black_mask: np.ndarray, dark_degree: int = 0):
        cur_ico = self.get_img(raw_rgb)
        ico_h, ico_w = cur_ico.shape[:2]

        bg_h, bg_w = block_img.shape[:2]

        start_row, end_row, start_col, end_col = tools.get_center_crop_pos((bg_h, bg_w), cropped_shape=(ico_h, ico_w))

        block_img_copy = block_img.copy()
        valid_pos = cur_ico[..., 3] != 0
        block_img_copy[start_row:end_row, start_col:end_col][valid_pos] = cur_ico[valid_pos]

        if dark_degree == 0:
            return block_img_copy
        block_img_copy.astype(np.float32)
        block_img_copy[..., :3] = block_img_copy[..., :3] / dark_degree

        return block_img_copy


def read_stuffs():
    img = './resource/img.png'
    img = cv2.imread(img, flags=cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    block_bg = Stuff('block_bg', (3, 3), wh=(120, 134))
    black_mask = Stuff('blackMask', (3, 144), wh=(120, 134))

    brush = Stuff('brush', (516, 641), wh=(94, 95))
    bucket = Stuff('bucket', (514, 199), wh=(95, 96), rotate=True)
    feeder = Stuff('feeder', (523, 846), wh=(96, 62), rotate=True)
    carrot = Stuff('carrot', (803, 286), wh=(92, 92))
    corn = Stuff('corn', (717, 714), wh=(91, 93))
    grass = Stuff('grass', (702, 511), wh=(89, 98), rotate=True)
    bell = Stuff('bell', (871, 3), wh=(96, 86))
    glove = Stuff('glove', (765, 3), wh=(100, 88))
    stump = Stuff('stump', (797, 500), wh=(87, 98), rotate=True)
    firewood = Stuff('firewood', (648, 910), wh=(74, 104))
    white_cloud = Stuff('white_cloud', (522, 301), wh=(94, 96))
    fork = Stuff('fork', (555, 3), wh=(99, 92), rotate=True)
    scissor = Stuff('scissor', (452, 3), wh=(97, 94), rotate=True)
    wool_ball = Stuff('wool_ball', (711, 615), wh=(92, 93), rotate=True)
    fire = Stuff('fire', (510, 530), wh=(86, 105), )
    pine = Stuff('pine', (127, 3), pt2=(234, 106), )
    claw = Stuff('pine', (3, 284), pt2=(105, 388), )
    onion = Stuff('onion', (870, 96), pt2=(964, 186), )

    ico_dict = {
        'brush': brush,
        'bucket': bucket,
        'feeder': feeder,
        'carrot': carrot,
        'corn': corn,
        'grass': grass,
        'bell': bell,
        'glove': glove,
        'stump': stump,
        'firewood': firewood,
        'white_cloud': white_cloud,
        'fork': fork,
        'scissor': scissor,
        'wool_ball': wool_ball,
        'fire': fire,
        'pine': pine,
        'claw': claw,
        'onion': onion,
    }

    for block in ico_dict:
        ico_dict[block].bright_img = ico_dict[block].get_bright_img(rgb.copy(), block_bg.get_img(rgb), black_mask.get_img(rgb), dark_degree=0)
        ico_dict[block].bright_dark_img = ico_dict[block].get_bright_img(rgb.copy(), block_bg.get_img(rgb), black_mask.get_img(rgb), dark_degree=2)
        ico_dict[block].dark_img = ico_dict[block].get_bright_img(rgb.copy(), block_bg.get_img(rgb), black_mask.get_img(rgb), dark_degree=3)
    return ico_dict


if __name__ == '__main__':
    read_stuffs()
