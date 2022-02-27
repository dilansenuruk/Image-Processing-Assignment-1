from ast import increment_lineno
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img1 = cv.imread(r"C:\Users\User\Desktop\ML\images\emma_gray.jpg", cv.IMREAD_GRAYSCALE)
assert img1 is not None

t1 = np.linspace(0, 50, 51)
t2 = np.linspace(100, 255, 100)
t3 = np.linspace(150, 255, 105)

t = np.concatenate((t1, t2, t3), axis = 0).astype(np.uint8)
plt.subplots(1, 1, figsize = (8, 4))
plt.title("Intensity Transformation")
plt.grid("on")
plt.plot(t)

assert len(t) == 256
img1_t = cv.LUT(img1, t)

cv.namedWindow("Image", cv.WINDOW_AUTOSIZE)
cv.imshow("Image", img1)
cv.waitKey(1000)
cv.imshow("Image", img1_t)
cv.waitKey(1000)
cv.destroyAllWindows()

fig, ax = plt.subplots(1, 2, figsize = (12, 6))

ax[0].imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(cv.cvtColor(img1_t, cv.COLOR_BGR2RGB))
ax[1].set_title("Intensity Transformed")
ax[1].axis("off")

plt.show()