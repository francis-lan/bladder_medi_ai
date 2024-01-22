import cv2
import numpy as np
from ultralytics import YOLO


def FF_ver_self(img, seed_point, threshold,y1,y2):
    match_points=[]
    seed_color = img[seed_point[1],seed_point[0]]
    for y in range(seed_point[1]-20,y2):
        current_color = img[y, seed_point[0]] 
        color_diff = (abs(int(current_color[0]) - int(seed_color[0])), abs(int(current_color[1]) - int(seed_color[1])), abs(int(current_color[2]) - int(seed_color[2])))
        if all(diff <= threshold for diff in color_diff):
            match_points.append((seed_point[0], y))
        
    print(seed_color)
    print(img[match_points[len(match_points)-1][1],match_points[len(match_points)-1][0]])
    return match_points

    
threshold = 50
x1, y1, x2, y2 = 100, 0, 700, 600
model = YOLO("C:/vscode.ai/runs/detect/train7/weights/best.pt")

img = cv2.imread("D:/train2/002.jpg")
img = cv2.resize(img, (900, 720))                             #限制大小
cropped_frame = img[y1:y2, x1:x2]                             #裁切需要範圍
black_background = np.zeros((720, 900, 3), dtype=np.uint8)    #建立黑底
black_background[y1:y2, x1:x2] = cropped_frame                #搞上去
img  = black_background



setxyxy = model.predict(img)

for r in setxyxy:
    boxes = r.boxes
    for box in boxes:
            
        b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format 左上跟右下端點的座標
        c = box.cls
bx1 = int(b[0]) + 20
by1 = int(b[1]) + 20    
bx2 = int(b[2]) - 20
by2 = int(b[3]) - 20
cv2.circle(img, (bx1, by1), 5, (0, 0, 255), -1)
cv2.circle(img, (bx2, by2), 5, (0, 0, 255), -1)
cv2.circle(img, (bx1, by2), 5, (0, 0, 255), -1)
cv2.circle(img, (bx2, by1), 5, (0, 0, 255), -1)
cv2.line(img, (bx1, by1), (bx1, by2), (0, 0, 255), 2)
cv2.line(img, (bx1, by1), (bx2, by1), (0, 0, 255), 2)
cv2.line(img, (bx2, by2), (bx2, by1), (0, 0, 255), 2)
cv2.line(img, (bx2, by2), (bx1, by2), (0, 0, 255), 2)

seed_point = ((bx1 + bx2)//2, (by1+by2)//2)  # 选取种子点

fill_color = (0, 255, 0)
fill_img = img.copy()
floodfill = cv2.floodFill(fill_img, None, seed_point, fill_color, (15, 15, 15), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
match_points = FF_ver_self(img, seed_point, threshold,by1,by2)
for point in match_points:
    cv2.line(img, seed_point, point, (0, 0, 255), 2)


cv2.circle(fill_img, seed_point, 5, (0, 0, 255), -1)
cv2.imshow('floodfill', fill_img)
cv2.imshow('img', img)
print(match_points)
wait = cv2.waitKey(0)