import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys 

tmp_w = 80
tmp_h = 110
tmp_def_w = 108
tmp_def_h = 150



def Perspective(filename,out_filename,src_pts,dts_pts):
    img = cv2.imread(filename)
    rows,cols,ch = img.shape #rows;画像の高さ、cols;画像の幅

    src_pts = np.float32(src_pts) #前の画像における4点の位置 [左上][右上][左下][右下]
    dts_pts = np.float32(dts_pts) #後の画像における4点の位置

    M = cv2.getPerspectiveTransform(src_pts,dts_pts)

    dst = cv2.warpPerspective(img,M,(tmp_w,tmp_h))

    src_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    dst_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB) 

    plt.subplot(121),plt.imshow(src_rgb),plt.title('input')
    plt.subplot(122),plt.imshow(dst_rgb),plt.title('output')
    plt.show()

    cv2.imwrite(out_filename,dst_rgb)

def getTmpPosition():
    tmp = []
    loc = [[[0]*2]*4]*10
    loc0 = [6,12] # 0の左上の位置
    
    for i in range(10):
        h = i//5
        loc[i][0]=[loc0[0]+i%5*tmp_def_w, loc0[1]+h*tmp_def_h] #左上
        loc[i][1]=[loc[i][0][0]+tmp_w,loc[i][0][1]]      #右上
        loc[i][2]=[loc[i][0][0]      ,loc[i][0][1]+tmp_h]#左下
        loc[i][3]=[loc[i][0][0]+tmp_w,loc[i][0][1]+tmp_h]#右下

        pts =[[0,0],[tmp_w,0],[0,tmp_h],[tmp_w,tmp_h]]
        Perspective('img/7segment_template.png','img/'+str(i)+'img.png',loc[i],pts)

    print(loc[5][0])
if __name__== '__main__':
    getTmpPosition()
