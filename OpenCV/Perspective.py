import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys 

# 591 1496
# 580 2067
# 2299 1353
# 2297 1919

def Perspective(filename):
    img = cv2.imread(filename)
    rows,cols,ch = img.shape #rows;画像の高さ、cols;画像の幅

    src_pts = np.float32([[590,1496],[2280,1380],[570,2070],[2300,1920]]) #前の画像における4点の位置 [左上][右上][左下][右下]
    dts_pts = np.float32([[0,0],[1700,0],[0,500],[1700,500]]) #後の画像における4点の位置

    M = cv2.getPerspectiveTransform(src_pts,dts_pts)

    dst = cv2.warpPerspective(img,M,(1700,500))

    src_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    dst_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB) 

    plt.subplot(121),plt.imshow(src_rgb),plt.title('input')
    plt.subplot(122),plt.imshow(dst_rgb),plt.title('output')
    plt.show()

    cv2.imwrite("clipped_image.png",dst_rgb)

if __name__== '__main__':
    filename = sys.argv[1]
    Perspective(filename)
