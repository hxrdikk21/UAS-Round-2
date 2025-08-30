import numpy as np
import cv2

img =cv2.imread('task_images/1.png', cv2.IMREAD_COLOR)
px = img[50,50]
img[100:150,100:150] = [255,0,255]
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()