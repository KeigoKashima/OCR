import cv2
import numpy as np
from matplotlib import pyplot as plt

def templateMaching(img,tmp):

    img2 = img.copy()
    w,h = tmp.shape[::-1]

    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',\
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        img = img2.copy()
        # eval関数は引数の関数を実行する
        method = eval(meth)

        # テンプレートマッチングを適用する
        responce = cv2.matchTemplate(img,tmp,method)

        threshold = 0.8
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(responce)

        # もしメソッドが TM_CCOEFF か、TM_SQDIFFなら、最小値を取得する。
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img,top_left,bottom_right,255,2)

        plt.subplot(121),plt.imshow(responce,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]),plt.yticks([])

        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'),plt.xticks([]),plt.yticks([])
        plt.suptitle(meth)

        plt.show()


def templateMaching_multi(img,tmp):

    img2 = img.copy()
    w,h = tmp.shape[::-1]

    # テンプレートマッチングを適用する
    responce = cv2.matchTemplate(img2,tmp,cv2.TM_CCOEFF_NORMED)

    threshold = 0.8
    loc = np.where(responce >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(img,pt,(pt[0]+w, pt[1]+h),(255),1)

    cv2.imshow('output.png',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # plt.plot(img)
    # plt.show()
 

if __name__ == "__main__":
    img = cv2.imread('gaussian_thresh_img.png',0)
    # tmp = cv2.imread('0_thresh_img_clipped.png',0)
    tmp = cv2.imread('img/remake_0img.png',0)
    tmp = cv2.bitwise_not(tmp)
    cv2.imshow('img',tmp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    templateMaching_multi(img,tmp)

