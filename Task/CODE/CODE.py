import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

# --- CONFIG: Priority, colors and capacities ---
shape_priority = {'star': 3, 'triangle': 2, 'square': 1}
emergency_priority = {'red': 3, 'yellow': 2, 'green': 1}
camp_colors_bgr = {'blue': (255, 203, 162), 'pink': (210, 183, 255), 'grey': (214, 214, 214)}
camp_capacity = {'blue': 4, 'pink': 3, 'grey': 2}

# --- Load image ---
img = cv2.imread("task_images/1.png")
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





#

# --- Detect camps (large rescue pads) ---
pad_mask = cv2.inRange(hsv, (0, 0, 150), (255, 255, 255))
contours, _ = cv2.findContours(pad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
camps = []
for cnt in contours:
    ((x, y), r) = cv2.minEnclosingCircle(cnt)
    if r > 20:
        px, py = int(x), int(y)
        bgr_c = tuple(img[py, px])
        camp_color = None
        for key, val in camp_colors_bgr.items():
            if np.allclose(val, bgr_c, atol=30):
                camp_color = key
                break
        if camp_color:
            camps.append({'pos': (px, py), 'color': camp_color, 'capacity': camp_capacity[camp_color], 'assigned': []})
print(f"Detected {len(camps)} camps:")
for camp in camps:
    print(f"  Camp {camp['color']} at {camp['pos']} with capacity {camp['capacity']}")

# --- Build color masks for casualties ---
yellow_mask = cv2.inRange(hsv, (20, 150, 150), (35, 255, 255))
lower_red = cv2.inRange(hsv, (0, 150, 150), (10, 255, 255))
upper_red = cv2.inRange(hsv, (170, 150, 150), (180, 255, 255))
red_mask = cv2.bitwise_or(lower_red, upper_red)
green_mask = cv2.inRange(hsv, (65, 150, 150), (85, 255, 255))
pink_mask = cv2.inRange(hsv, (140, 100, 180), (170, 255, 255))
casualty_mask = cv2.bitwise_or(yellow_mask, red_mask)
casualty_mask = cv2.bitwise_or(casualty_mask, green_mask)
casualty_mask = cv2.bitwise_or(casualty_mask, pink_mask)
cv2.imwrite("casualty_mask_debug.png", casualty_mask)

# --- Find contours on combined casualty mask ---
contours, _ = cv2.findContours(casualty_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Contours found for casualties: {len(contours)}")

# --- Helper for shape classification ---
def classify_shape(cnt):
    approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
    sides = len(approx)
    # For stars, sides might be ~10 or more, so allow len >= 9 for star
    if sides >= 9:
        return 'star'
    elif sides == 3:
        return 'triangle'
    elif sides == 4:
        return 'square'
    return None

# --- Process contours to get casualties ---
casualties = []
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < 5 or area > 4000:
        continue
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue
    px = int(M["m10"] / M["m00"])
    py = int(M["m01"] / M["m00"])
    bgr = img[py, px]
    shape = classify_shape(cnt)
    # Identify emergency color by BGR color near centroid
    b, g, r = bgr
    if r > 200 and g < 90 and b < 90:
        ec = 'red'
    elif r > 230 and g > 210 and b < 120:
        ec = 'yellow'
    elif g > 200 and r < 130 and b < 120:
        ec = 'green'
    elif r > 180 and b > 180 and g < 160:
        ec = 'pink'  # rescue pads, ignore if detected here
    else:
        ec = 'yellow'  # fallback to yellow
    if shape is not None and ec in emergency_priority:
        casualties.append({
            'pos': (px, py),
            'shape': shape,
            'emergency': ec,
            'casualty_prio': shape_priority[shape],
            'emergency_prio': emergency_priority[ec],
            'priority': shape_priority[shape]*emergency_priority[ec]
        })
print(f"Detected {len(casualties)} casualties:")
for c in casualties:
    print(f"  Pos {c['pos']}, Shape={c['shape']}, Emergency={c['emergency']}, Priority={c['priority']}")

# --- Assign casualties to camps ---
def assignment_score(cas, camp):
    dist = np.linalg.norm(np.subtract(cas['pos'], camp['pos']))
    return cas['priority'] * 1000 - dist

remain = casualties.copy()
for camp in camps:
    selection = []
    if not remain:
        continue
    ranked = sorted(remain, key=lambda c: assignment_score(c, camp), reverse=True)
    for cas in ranked[:camp['capacity']]:
        camp['assigned'].append(cas)
        selection.append(cas)
    remain = [r for r in remain if r not in selection]

# --- Format outputs ---
output_details = []
camp_count = []
camp_priority = []
total_priority = 0
num_casualties = sum(len(c['assigned']) for c in camps)

for camp in camps:
    dets = []
    score_sum = 0
    for cas in camp['assigned']:
        dets.append([cas['casualty_prio'], cas['emergency_prio']])
        score_sum += cas['priority']
    output_details.append(dets)
    camp_count.append(len(camp['assigned']))
    camp_priority.append(score_sum)
    total_priority += score_sum

rescue_ratio = total_priority / num_casualties if num_casualties else 0

# --- Print results ---
print("Segmented output image: segmented_1.jpg")
print(f"Image: 1.png\n")
for camp in camps:
    print(f"Camp: {camp['color'].capitalize()} at {camp['pos']} (Capacity: {camp['capacity']})")
    if camp['assigned']:
        for i, cas in enumerate(camp['assigned'], 1):
            print(f"  {i}. Casualty at {cas['pos']}: Shape={cas['shape']}, Emergency={cas['emergency']}, Priority={cas['priority']}")
        print(f"  Camp total priority: {sum(cas['priority'] for cas in camp['assigned'])}")
    else:
        print("  No casualties assigned.")
    print()
print(f"Total casualties assigned: {num_casualties}")
print(f"Total priority score for image: {total_priority}")
print(f"Rescue ratio Pr: {rescue_ratio:.2f}")



plt.figure(figsize=(8, 8))
plt.imshow(casualty_mask, cmap='gray')
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    plt.gca().add_patch(plt.Rectangle((x,y), w, h, edgecolor='r', fill=False, linewidth=1))
plt.title('Casualty Mask and Detected Contours')
plt.axis('off')
plt.show()
# 6) Save or show
cv2.imwrite("segmented1.png", final_vis) 
cv2.imshow("overlay", final_vis);
cv2.imshow("original", img);
cv2.waitKey(0); cv2.destroyAllWindows()