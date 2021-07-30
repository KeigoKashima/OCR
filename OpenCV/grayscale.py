import cv2
import numpy as np
import matplotlib.pyplot as plt

img_color = cv2.imread('test.png',cv2.IMREAD_COLOR)
img_grayscale = cv2.imread('test.png',cv2.IMREAD_GRAYSCALE)
img_unchanged = cv2.imread('test.png',cv2.IMREAD_UNCHANGED)

## openCVで表示する。
# cv2.imshow('image',img_grayscale)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# matplotlibで表示
# OpenCVではBGR順、matplotlibではRGB順。
# なので、cvtColorで変換する
img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB) # BGR -> RGB順に
img_grayscale = cv2.cvtColor(img_grayscale, cv2.COLOR_BGR2RGB) # BGR -> RGB順に
img_unchanged = cv2.cvtColor(img_unchanged, cv2.COLOR_BGR2RGB) # BGR -> RGB順に

plt.subplot(131),plt.imshow(img_color),plt.title('color')
plt.subplot(132),plt.imshow(img_grayscale),plt.title('grayscale')
plt.subplot(133),plt.imshow(img_unchanged),plt.title('unchange')
plt.show()

cv2.imwrite("img_grayscale.png",img_grayscale)
    