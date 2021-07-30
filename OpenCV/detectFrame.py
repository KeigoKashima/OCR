import cv2 
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.shape_base import apply_over_axes


def draw_contours(img, contours, ax):
    """
        輪郭の点及び線を画像上に描画する。
        ＜引数＞
        img:元の画像
        contours:輪郭の頂点
        ax:画像サイズ
    """
    ax.imshow(img)
    ax.set_axis_off()

    for i, cnt in enumerate(contours):
        # 形状を変更する。(NumPoints, 1, 2) -> (NumPoints, 2)
        cnt = cnt.squeeze(axis=1)
        # 輪郭の点同士を結ぶ線を描画する。
        ax.add_patch(plt.Polygon(cnt, color="b", fill=None, lw=2))
        # 輪郭の点を描画する。
        ax.plot(cnt[:, 0], cnt[:, 1], "ro", mew=0, ms=4)
        # 輪郭の番号を描画する。
        ax.text(cnt[0][0], cnt[0][1], i, color="r", size="20", bbox=dict(fc="w"))


def findRectangle(img):
    """
        長方形の輪郭を探す関数
        ＜引数＞
        img:元の画像
    """

    contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # 小さい輪郭は誤検出として削除する
    contours = list(filter(lambda x: cv2.contourArea(x) > 100000, contours))

    # 近似の輪郭
    approx_contours = []
    for i, cnt in enumerate(contours):
        # 輪郭の周囲の長さを計算する。
        arclen = cv2.arcLength(cnt, True)
        # 輪郭を近似する。
        approx_cnt = cv2.approxPolyDP(cnt, epsilon=0.005 * arclen, closed=True)
        approx_contours.append(approx_cnt)

    # 四角形を探す
    # filter関数の中にlambda式を使うことで、条件に合うものを抽出している。
    # ここでは、approx_contoursの要素の中でリストの長さが４のもの、つまり頂点が４のものを抽出
    rectangles = list(filter(lambda x: len(x) == 4, approx_contours))

    # 描画用
    fig, ax = plt.subplots(figsize=(8, 8))

    # cv2ではBGRなので、RGBに変換
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB順に

    draw_contours(img, rectangles, ax)
    print(rectangles)

    plt.show()

    return rectangles


def detectFrame(img,out):

    contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # print(hierarchy)

    # 小さい輪郭は誤検出として削除する
    contours = list(filter(lambda x: cv2.contourArea(x) > 10000, contours))

    external_contours = np.zeros(img.shape)

    for i in range(len(contours )):
        # 一番外側の輪郭を表示する。
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(external_contours, contours, i,255, thickness=3)
            print(hierarchy[0][i])
    

    cv2.imshow('oroginal',img)
    cv2.waitKey(0)
    cv2.imshow("color",external_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # output = cv2.drawContours(img, contours, -1, (0,0,255), thickness=3)

    
    # plt.subplot(121),plt.imshow(img),plt.title('original')
    # # # plt.subplot(132),plt.imshow(image),plt.title('pre')
    # plt.subplot(122),plt.imshow(external_contours),plt.title('grayscale')

    # plt.show()




if __name__ == '__main__':
    filename = sys.argv[1]
    output = sys.argv[2]
    img = cv2.imread(filename,0)
    inv = cv2.bitwise_not(img)
    # detectFrame(inv,output)
    findRectangle(img)