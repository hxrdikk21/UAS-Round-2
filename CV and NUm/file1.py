import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)
#IMREAD_COLOR - 1
#IMREAD_UNCHANGED - -1
#IMREAD_GRAYSCALE - 0

cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.plot([50,100],[100,50],'r',linewidth=4)
plt.show()"""
cv2.imwrite('watchgray.png',img)
