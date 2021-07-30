import cv2
import numpy as np

img = cv2.imread('test.png',0)
rows,cols = img.shape #rows;画像の高さ、cols;画像の幅

#  (t_x,t_y)=(100,50)移動させる 
M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img,M,(cols,rows)) #第３引数（width,height）

cv2.imshow('img.png',dst)
cv2.waitKey(0) #0を入力すると終了する
cv2.destroyAllWindows() #現在までに作られた全てのウインドウを閉じる