import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('task_images/6.png',cv2.IMREAD_COLOR)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

color1 =[255,0,0]   #BLUE for water
color2 =[0,255,255]   #YELLOW for land]
alpha = 0.45

overlay1 = img.copy()
overlay2 = img.copy()

# Ocean (dark blue background)
lower_ocean = np.array([95, 60, 30])
upper_ocean = np.array([130, 255, 200])
mask_ocean = cv2.inRange(hsv, lower_ocean, upper_ocean)  # [12][3]

# Land (green coast)
lower_land = np.array([35, 40, 40])
upper_land = np.array([85, 255, 255])
mask_land = cv2.inRange(hsv, lower_land, upper_land)

# 5) Paint where masks are 255, then blend with original
overlay = img.copy()
overlay[mask_ocean == 255] = color1
blend = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0.0)  

overlay2 = blend.copy()
overlay2[mask_land == 255]  = color2
final_vis = cv2.addWeighted(overlay2, alpha, blend, 1 - alpha, 0.0)  

# 6) Save or show
cv2.imwrite("segmented6.png", final_vis) 
cv2.imshow("overlay", final_vis);
cv2.imshow("original", img);
cv2.waitKey(0); cv2.destroyAllWindows()