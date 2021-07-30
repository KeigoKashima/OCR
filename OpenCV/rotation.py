import cv2
import numpy as np

img = cv2.imread('test.png',0)
rows,cols = img.shape #rows;画像の高さ、cols;画像の幅

M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1) #１回転中心、２回転角度、3スケール
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow('img.png',dst)
cv2.waitKey(0) #0を入力すると終了する
cv2.destroyAllWindows() #現在までに作られた全てのウインドウを閉じる