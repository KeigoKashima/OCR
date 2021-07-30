import cv2
import numpy as np 
from matplotlib import pyplot as plt
import sys
import math
import os

###### 二値化 #####################

def binarization(img):
    """
    大津の二値化を使う
    双峰性を持つヒストグラムを持つ画像に対して、二つのピークの間を閾値とする。
    ＜引数＞
    img：元の画像
    """
    # # cv2.threshold(入力画像(グレースケール),閾値,最大値,閾値処理方法)
    # # 返り値　閾値,二値画像
    # ガウジアンフィルタ後に、大津の二値化を適用する
    # cv2.GaussianBlur(画像、フィルタのサイズ)
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imwrite("debug/binalization.png",thresh)

    return thresh


###### 枠の検出 #####################

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
    # 輪郭で囲われている領域の面積でフィルターをかける。
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

    # draw_contours(img, rectangles, ax)
    # print(rectangles)
    # plt.show()

    return rectangles


###### 幾何変換　#####################

def perspective(img,rectangles):
    """
    　透視射影変換する関数
    　＜引数＞
    　img:元の画像
    　rectangles:枠の頂点
    """
    # img = cv2.imread(filename)
    rows,cols= img.shape #rows;画像の高さ、cols;画像の幅

    # src_pts = np.float32(rectangles)
    src_pts = np.float32([rectangles[3],rectangles[0],rectangles[2],rectangles[1]]) #前の画像における4点の位置 [左上][右上][左下][右下]
    dts_pts = np.float32([[0,0],[1700,0],[0,500],[1700,500]]) #後の画像における4点の位置

    print(src_pts)
    print(dts_pts)
    M = cv2.getPerspectiveTransform(src_pts,dts_pts)

    dst = cv2.warpPerspective(img,M,(1700,500))

    return dst

    # cv2.imwrite("clipped_image.png",dst_rgb)

def skew(img,angle):
    """
    　指定した角度(angle)だけ画像を傾ける関数。
    　＜引数＞
    　img:元の画像
    　angle:傾ける角度（反時計まわり）
    """
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

    return affine_img_skew_x

###### OCR #####################




if __name__ == '__main__':
    # 画像の取得
    filename = sys.argv[1]
    img = cv2.imread(filename,0)
    # 二値化
    thresh = binarization(img)
    # 枠の検出
    rectangles = findRectangle(thresh)
    
    #　幾何変換
    # rはrectanglesの要素
    for i in range(len(rectangles)):
        dst = perspective(thresh,rectangles[i])
        #　小数点を認識しやすいように、傾ける。
        skewed = skew(dst,10)

        # 画像を出力する
        output = "output/out"+str(i)+".png"
        print(output)
        cv2.imwrite(output,skewed)
        # OCR
        cmd = 'ssocr crop 240 0  1450 480 invert -r4 -d-1 shear 100 '+ str(output) +' -D'
        os.system(cmd)

    

    

