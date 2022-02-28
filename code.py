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







img2 = cv.imread(r"C:\Users\User\Desktop\ML\images\brain_proton_density_slice.png", cv.IMREAD_GRAYSCALE)
assert img2 is not None

t1 = np.linspace(10, 10, 216)
t2 = np.linspace(200, 200, 40)
t3 = np.linspace(10, 10, 0)
t = np.concatenate((t1, t2, t3), axis = 0).astype(np.uint8)

r1 = np.linspace(10, 10, 186)
r2 = np.linspace(200, 200, 30)
r3 = np.linspace(10, 10, 40)
r = np.concatenate((r1, r2, r3), axis = 0).astype(np.uint8)

assert len(t) == 256
img2_t = cv.LUT(img2, t)

assert len(r) == 256
img2_r = cv.LUT(img2, r)

cv.namedWindow("Image", cv.WINDOW_AUTOSIZE)
cv.imshow("Image", img2)
cv.waitKey(1000)
cv.imshow("Image", img2_t)
cv.waitKey(1000)
cv.imshow("Image", img2_r)
cv.waitKey(1000)
cv.destroyAllWindows()

fig, ax = plt.subplots(1, 3, figsize = (18, 6))

ax[0].imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(cv.cvtColor(img2_t, cv.COLOR_BGR2RGB))
ax[1].set_title("White Matter")
ax[1].axis("off")

ax[2].imshow(cv.cvtColor(img2_r, cv.COLOR_BGR2RGB))
ax[2].set_title("Gray Matter")
ax[2].axis("off")

fig, ax = plt.subplots(1, 2, figsize = (18, 6))
ax[0].plot(t)
ax[0].set_title("Intensity Transformation - White Matter")
ax[0].grid("on")

ax[1].plot(r)
ax[1].set_title("Intensity Transformation - Gray Matter")
ax[1].grid("on")

plt.show()







img3 = cv.imread(r"C:\Users\User\Desktop\ML\images\highlights_and_shadows.jpg", cv.IMREAD_COLOR)
assert img3 is not None

cv.namedWindow("Image", cv.WINDOW_AUTOSIZE)
cv.imshow("Image", img3)
cv.waitKey(1000)
cv.destroyAllWindows()

gamma = [0.2, 0.8, 1.2, 2.0]
hist_img3_lab = []

lab = cv.cvtColor(img3, cv.COLOR_BGR2LAB)

for i in gamma:
    k = np.array([(p/255)**(i)*255 for p in range(0, 256)]).astype(np.uint8)
    for x in range(0, len(lab)):
        for y in range(0, len(lab[0])):
            lab[x, y][0] = ((lab[x, y][0] / 255)**(i)) * 255
    
    hist_img3_lab.append(cv.calcHist([lab], [0], None, [256], [0, 256]))
    
    fig, ax = plt.subplots(1, 2, figsize = (12, 6))
    ax[0].plot(k)
    ax[0].set_title("L* Plane Curve (Gamma = " + str(i) + ")")
    ax[0].grid("on")
    
    ax[1].imshow(cv.cvtColor(lab, cv.COLOR_LAB2RGB))
    ax[1].set_title("Gamma Corrected (Gamma = " + str(i) + ")")
    ax[1].axis("off")







hist_img3 = cv.calcHist([img3], [0], None, [256], [0, 256])
fig, ax = plt.subplots(1, 2, figsize=(18, 4))
ax[0].imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
ax[0].axis("off")
ax[0].set_title("Original")

ax[1].plot(hist_img3)
ax[1].set_title("Histogram for Original Image")

fig, ax = plt.subplots(2, 2, figsize=(16, 8))
k = 0
f = 0
j = 0
for i in hist_img3_lab:
    ax[k, f].plot(i)
    ax[k, f].set_title("Histogram for Gamma Corrected Image (Gamma = " + str(gamma[j]) + ")")
    f += 1
    if (f == 2):
        k += 1
        f = 0
    j += 1

plt.show()