import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

# 単純な閾値処理
def simple_binarization():
    # cv2.threshold(入力画像(グレースケール),閾値,最大値,閾値処理方法)
    # 返り値　閾値,二値画像
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

    titles = ['origianl','binary','binary_inv','trunc','tozero','tozero_inv']
    images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]

    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()

# 適応的閾値処理
# 画像中の小領域ごとに閾値を計算する
def adaptive_binarization(img):
    # 平滑化
    img = cv2.medianBlur(img,5)

    # cv2.threshold(入力画像(グレースケール),閾値,最大値,閾値処理方法)
    # 返り値　retval,二値画像
    ret,thresh1 = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
    
    '''
    cv2.adaptiveThreshold(入力画像(グレー),最大値,)
        １：入力画像
        ２：最大値
        ３：閾値の計算方法
        ４：閾値の種類
        ５：閾値計算に利用する近傍サイズ（３なら８近傍）
        ６：計算した閾値かrあCを引いた値を最終的な閾値にする
    '''
    # cv2.ADAPTIVE_THRESH_MEAN_C；近傍領域の中央値を閾値とします
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C：近傍領域の重み付け平均値を閾値とする。重みの値はGaussian分布になるように計算される。
    thresh2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    thresh3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    
    titles = ['origianl','global threshhold(v=100)','adaptive mean threshold','adaptive gaussian threshold']
    images = [img,thresh1,thresh2,thresh3]

    for i in range(4):
        plt.subplot(1,4,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()

# 大津の二値化
# 双峰性を持つヒストグラムを持つ画像に対して、二つのピークの間を閾値とする。

def Otsu_binarization(img,output):
    # cv2.threshold(入力画像(グレースケール),閾値,最大値,閾値処理方法)
    # 返り値　閾値,二値画像
    ret1,thresh1 = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
    
    # 大津の二値化
    # retval ＝ アルゴリズムで計算された閾値
    
    
    ret2,thresh2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5,5),np.uint8)
    thresh2 = cv2.erode(thresh2,kernel,iterations = 3)
    
    # ガウジアンフィルタ後に、大津の二値化を適用する
    # cv2.GaussianBlur(画像、フィルタのサイズ)
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,thresh3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    images = [
        img,0,thresh1,
        img,0,thresh2,
        blur,0,thresh3
    ]
    titles =['original','histogram','thresh v=100',
            'original','histogram',"otsu's thresh",
            'gaussian filtered image','histogram',"otsu's thresh"]
        
    for i in range(3):
        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3]),plt.xticks([]),plt.yticks([])

        plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1]),plt.xticks([]),plt.yticks([])

        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+2]),plt.xticks([]),plt.yticks([])

    plt.show()

    
    cv2.imwrite(output,thresh3)
    # cv2.imwrite('gaussian_thresh_img.png',thresh3)
    

if __name__ == '__main__':
    filename = sys.argv[1]
    output = sys.argv[2]
    
    img = cv2.imread(filename,0)
    
    # img_grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # BGR -> RGB順に
    # adaptive_binarization(img)
    Otsu_binarization(img,output)