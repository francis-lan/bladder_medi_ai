import cv2
import numpy as np
from ultralytics import YOLO

def FF_ver_self(img, seed_point, threshold,y1,y2):
    match_points=[]
    seed_color = (0, 0, 0)
    for y in range(seed_point[1]-20,y2):
        current_color = img[y, seed_point[0]] 
        color_diff = (abs(int(current_color[0]) - int(seed_color[0])), abs(int(current_color[1]) - int(seed_color[1])), abs(int(current_color[2]) - int(seed_color[2])))
        if all(diff <= threshold for diff in color_diff):
            match_points.append((seed_point[0], y))
        else:
            break
        
    
    return match_points

def FF_ver_top(img, seed_point, threshold, y1, y2):
    match_points=[]
    seed_color = (0, 0, 0)
    for y in range(seed_point[1], y1 - 1, -1):
        current_color = img[y, seed_point[0]] 
        color_diff = (abs(int(current_color[0]) - int(seed_color[0])), abs(int(current_color[1]) - int(seed_color[1])), abs(int(current_color[2]) - int(seed_color[2])))
        if all(diff <= threshold for diff in color_diff):
            match_points.append((seed_point[0], y))
        else:
            break
        
    
    return match_points

threshold = 70
model = YOLO("C:/vscode.ai/runs/detect/train7/weights/best.pt")
#cap = cv2.VideoCapture('D:/User-Data/Downloads/kegal_keep1.mp4')
#cap = cv2.VideoCapture('D:/User-Data/Downloads/kegal_2.mp4')
#cap = cv2.VideoCapture('D:/User-Data/Downloads/kegal_1.mp4')
cap = cv2.VideoCapture('D:/User-Data/Downloads/kegal_keep2.mp4')
#cap = cv2.VideoCapture("D:/User-Data/Downloads/1701334235.mp4")
#cap = cv2.VideoCapture("D:/User-Data/Downloads/1701332749.mp4")
#cap = cv2.VideoCapture("D:/User-Data/Downloads/1701332680.mp4")
#cap = cv2.VideoCapture('D:/User-Data/Downloads/TaUS_K1(kwT).mp4')
#cap = cv2.VideoCapture('D:/User-Data/Downloads/TaUS_V(Wrong).mp4')
#cap = cv2.VideoCapture(0)
frame_rate = int(cap.get(5))                                      #影片幀率
x1, y1, x2, y2 = 100, 0, 700, 600                                 #剪裁範圍
gap = []
point = []
huge_gap = []
#之後鏡頭輸入用
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    
    img = cv2.resize(img, (900, 720))                             #限制大小
    cropped_frame = img[y1:y2, x1:x2]                             #裁切需要範圍
    black_background = np.zeros((720, 900, 3), dtype=np.uint8)    #建立黑底
    black_background[y1:y2, x1:x2] = cropped_frame                #搞上去
    img  = black_background
    
    #results = model(img)                      #yolo直接預測
    setxyxy = model.predict(img)

    for r in setxyxy:
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format 左上跟右下端點的座標
            c = box.cls
    bx1 = int(b[0]) + 20
    by1 = int(b[1]) + 20    
    bx2 = int(b[2]) - 20
    by2 = int(b[3]) 
   
    cv2.line(img, (bx1, by1), (bx1, by2), (0, 0, 255), 2)
    cv2.line(img, (bx1, by1), (bx2, by1), (0, 0, 255), 2)
    cv2.line(img, (bx2, by2), (bx2, by1), (0, 0, 255), 2)
    cv2.line(img, (bx2, by2), (bx1, by2), (0, 0, 255), 2)

    seed_point = ((bx1 + bx2)//2, (by1+by2)//2)
    left_point = (((bx1+bx2)//2 + bx1)//2, (by1+by2)//2)
    right_point = (((bx1+bx2)//2 + bx2)//2, (by1+by2)//2)


    match_points = FF_ver_self(img, seed_point, threshold,by1,by2)
    left_match_points = FF_ver_self(img, left_point, threshold,by1,by2)
    right_match_points = FF_ver_self(img, right_point, threshold,by1,by2)
    left_top_match_points = FF_ver_top(img, left_point, threshold,by1,by2)
    right_top_match_points = FF_ver_top(img, right_point, threshold,by1,by2)
    bugap = [0, 0]


    if len(left_top_match_points) > 0:
        cv2.line(img, left_point, left_top_match_points[len(left_top_match_points)-1], (255, 0, 0), 2)
    elif len(right_top_match_points) > 0:
        cv2.line(img, right_point, right_top_match_points[len(right_top_match_points)-1], (255, 0, 0), 2)


    if len(right_match_points) > 0:
        cv2.line(img, right_point, right_match_points[len(right_match_points)-1], (0, 255, 0), 2)
    else:
        cv2.circle(img, right_point, 5, (0, 0, 255), -1)
    if len(left_match_points) > 0:
        cv2.line(img, left_point, left_match_points[len(left_match_points)-1], (0, 255, 0), 2)
    else:
        cv2.circle(img, left_point, 5, (0, 0, 255), -1)
    


    if len(match_points) > 0 and len(left_match_points) > 0 and len(right_match_points) > 0 and len(left_top_match_points) > 0 and len(right_top_match_points) > 0:
        matchgap = [match_points[len(match_points)-1][1] - seed_point[1], left_match_points[len(left_match_points)-1][1] - left_point[1], right_match_points[len(right_match_points)-1][1] - right_point[1]]
        matchpow = [match_points[len(match_points)-1][1], left_match_points[len(left_match_points)-1][1], right_match_points[len(right_match_points)-1][1]]
        bugap = [left_match_points[len(left_match_points)-1][1] - left_top_match_points[len(left_top_match_points)-1][1], right_match_points[len(right_match_points)-1][1] - right_top_match_points[len(right_top_match_points)-1][1]]
    elif len(match_points) > 0 and len(left_match_points) > 0 and len(left_top_match_points) > 0:
        matchgap = [match_points[len(match_points)-1][1] - seed_point[1], left_match_points[len(left_match_points)-1][1] - left_point[1], 0]
        matchpow = [match_points[len(match_points)-1][1], left_match_points[len(left_match_points)-1][1], right_point[1]]
        
    elif len(match_points) > 0 and len(right_match_points) > 0:
        matchgap = [match_points[len(match_points)-1][1] - seed_point[1], 0, right_match_points[len(right_match_points)-1][1] - right_point[1]]
        matchpow = [match_points[len(match_points)-1][1], left_point[1], right_match_points[len(right_match_points)-1][1]]
        
    gap.append(matchgap)
    point.append(matchpow)
    huge_gap.append(bugap)
    cv2.imshow("img", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
print("gap = ", gap)
print("point = ", point)
print("huge_gap = ", huge_gap)
cap.release()
cv2.destroyAllWindows()