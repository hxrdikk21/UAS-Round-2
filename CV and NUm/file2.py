import numpy as np
import cv2

img = cv2.imread('watch.jpg', cv2.IMREAD_COLOR)
cv2.rectangle(img , (15,25) , (200,125) ,(255,255,25),5)

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True,(0,200,2000),5)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,"Lmaoo Kys",(100,100), font, 1, (255,0,255) ,2,cv2.LINE_AA)


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()