import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img_rgb = cv.imread('darktable_exported/DJI_0049.jpg')
assert img_rgb is not None, "file could not be read, check with os.path.exists()"
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('template.png', cv.IMREAD_GRAYSCALE)
assert template is not None, "file could not be read, check with os.path.exists()"
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
print(res)
threshold = 0.98
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    x = pt[0] + w/2
    y = pt[1] + h/2
    print(f"{x}, {y}")
cv.imwrite('res.png',img_rgb)
