import math
import numpy as np
from matplotlib import pyplot as plt
import cv2 
import sys

def skew(img,angle):
    a = math.tan(math.radians(angle))
    h,w=img.shape

    mat = np.array([[1, a, 0], [0, 1, 0]], dtype=np.float32)
    print(mat)
    # [[1.        0.2679492 0.       ]
    #  [0.        1.        0.       ]]

    affine_img_skew_x = cv2.warpAffine(img, mat, (int(w + h * a), h))
    # cv2.imwrite('data/dst/opencv_affine_skew_x.jpg', affine_img_skew_x)

    plt.subplot(121),plt.imshow(img),plt.title('input')
    plt.subplot(122),plt.imshow(affine_img_skew_x),plt.title('output')
    plt.show()

if __name__ == '__main__':
    # 画像の取得
    filename = sys.argv[1]
    img = cv2.imread(filename,0)

    skew(img,10)