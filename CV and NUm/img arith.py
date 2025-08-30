import numpy as np
import cv2

img1 = cv2.imread('task_images/1.png', cv2.IMREAD_COLOR)
img2 = cv2.imread('task_images/2.png', cv2.IMREAD_COLOR)

#sum = cv2.add(img1,img2) 
#sum2 = cv2.addWeighted(img1,0.5,img2,0.5,0)
#cv2.imshow('Sum',sum2)



img2gray =cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) 
ret, mask = cv2.threshold(img2gray,100,255,cv2.THRESH_BINARY_INV)
cv2.imshow('mask',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()