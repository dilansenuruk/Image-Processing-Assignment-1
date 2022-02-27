from ast import increment_lineno
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img1 = cv.imread(r"C:\Users\User\Desktop\ML\images\emma_gray.jpg", cv.IMREAD_GRAYSCALE)
assert img1 is not None

t1 = np.linspace(0, 50, 50)
t2 = np.linspace(50, 100, 0)
t3 = np.linspace(100, 255, 100)
t4 = np.linspace(255, 150, 0)
t5 = np.linspace(150, 255, 106)

t = np.concatenate((t1, t2, t3, t4, t5), axis = 0).astype(np.uint8)
plt.subplots(1, 1, figsize = (8, 4))
plt.plot(t)
plt.show()

assert len(t) == 256
img1_t = cv.LUT(img1, t)

cv.namedWindow("Image", cv.WINDOW_AUTOSIZE)
cv.imshow("Image", img1)
cv.waitKey(2000)
cv.imshow("Image", img1_t)
cv.waitKey(2000)
cv.destroyAllWindows()