import cv2

img = cv2.imread('img/7segment_template.png', cv2.IMREAD_COLOR)
window_name = 'img'

def onMouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

cv2.imshow(window_name, img)
cv2.setMouseCallback(window_name, onMouse)
cv2.waitKey(0)