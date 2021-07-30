import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt


# 画像を表示する関数
def display(img):
    # fig = plt.figure(figsize=(12,10))
    # ax = fig.add_subplot(111)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def SIFT_matching(img,tmp):
    # sift = cv2.xfeatures2d.SIFT_create() #SIFTの特許がきれて、下ので使えるようになったらしい。
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img,None)
    kp2, des2 = sift.detectAndCompute(tmp,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    good = []
    for match1,match2 in matches:
        if match1.distance < 0.75*match2.distance:
            good.append([match1])  

    sift_matches = cv2.drawMatchesKnn(img,kp1,tmp,kp2,good,None,flags=2)

    display(sift_matches)   

def FLANN_matching(img,tmp):

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img,None)
    kp2, des2 = sift.detectAndCompute(tmp,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)  

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    good = []

    for i,(match1,match2) in enumerate(matches):
        if match1.distance < 0.7*match2.distance:
            
            good.append([match1])


    flann_matches = cv2.drawMatchesKnn(img,kp1,tmp,kp2,good,None,flags=0)

    display(flann_matches)

if __name__ == "__main__":
    img = cv2.imread('gaussian_thresh_img.png',0)
    tmp = cv2.imread('img/remake_0img.png',0)
    tmp = cv2.bitwise_not(tmp) #反転


    # SIFT_matching(img,tmp)
    FLANN_matching(img,tmp)

    

