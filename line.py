import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
 
I = image.imread('IMG_1698.jpg') 
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

# Canny Edge Detection:
Threshold1 = 10;
Threshold2 = 100;
FilterSize = 10
E = cv2.Canny(G, Threshold1, Threshold2, FilterSize)

Rres = 1
Thetares = 1*np.pi/180
Threshold = 1
minLineLength = 1
maxLineGap = 100
lines = cv2.HoughLinesP(E,Rres,Thetares,Threshold,minLineLength,maxLineGap)

if lines is not None:
    N = lines.shape[0]
    for i in range(N):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]    
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]    
        cv2.line(I,(x1,y1),(x2,y2),(255,0,0),2)
else:
    print("No lines")

# a = cv2.inRange(I, 10, 255, a)


hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)

## mask of green (36,0,0) ~ (70, 255,255)
mask1 = cv2.inRange(hsv, (100, 100, 0), (100, 255,255))

## mask o yellow (15,0,0) ~ (36, 255, 255)
# mask2 = cv2.inRange(hsv, (15,0,0), (36, 255, 255))

## final mask and masked
# mask = cv2.bitwise_or(mask1, mask2)
target = cv2.bitwise_and(I,I, mask=mask1)


w=10
h=10
fig=plt.figure(figsize=(4, 4))

fig.add_subplot(1, 2, 1)
plt.imshow(I)

fig.add_subplot(2, 3, 2)
plt.imshow(target)

# columns = 4
# rows = 5
# for i in range(1, columns*rows +1):
#     img = np.random.randint(10, size=(h,w))
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(img)
plt.show()