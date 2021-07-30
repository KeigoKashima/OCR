"""
    画像のサイズ変更を行う。
    使用する関数
        cv2.resize()
        １：入力画像
        ２：サイズ
        ３：補完方法
"""

import cv2
import numpy as np

# read the imagefile
img = cv2.imread('img/0img.png')

height,width = img.shape[:2]
size = (480*80//110, 480)

res = cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC)

cv2.imwrite('img/resize_0img.png', res)
