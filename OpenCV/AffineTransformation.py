'''
アフィン変換；回転させたり、引き伸ばす変換

変換前後で並行性を保つ変換。

'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test.png',0)
rows,cols = img.shape #rows;画像の高さ、cols;画像の幅

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('input')
plt.subplot(122),plt.imshow(dst),plt.title('output')
plt.show()
