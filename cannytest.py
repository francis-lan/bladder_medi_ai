import cv2
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO


model = YOLO("C:/vscode.ai/runs/detect/train7/weights/best.pt")

# 读取图像
cap = cv2.VideoCapture('D:/User-Data/Downloads/kegal_keep1.mp4')

frame_rate = int(cap.get(5))                                      #影片幀率
x1, y1, x2, y2 = 100, 0, 700, 600                                 #剪裁範圍

#之後鏡頭輸入用
if not cap.isOpened():
    print("Cannot open camera")
    exit()

#開始操作
while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    #對輸入影像做基本處裡
    img = cv2.resize(img, (900, 720))                             #限制大小
    cropped_frame = img[y1:y2, x1:x2]                             #裁切需要範圍
    black_background = np.zeros((720, 900, 3), dtype=np.uint8)    #建立黑底
    black_background[y1:y2, x1:x2] = cropped_frame                #搞上去
    img  = black_background    
    fill_img = img.copy()
    setxyxy = model.predict(img)
    ret, thresholded = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
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
    seed_point = ((bx1 + bx2)//2, (by1+by2)//2)  # 选取种子点
    fill_color = (0, 255, 0)
    floodfill = cv2.floodFill(fill_img, None, seed_point, fill_color, (15, 15, 15), (30, 30, 30), cv2.FLOODFILL_FIXED_RANGE)
# 使用Canny边缘检测
    edges = cv2.Canny(img, 70, 200)  # 参数分别是低阈值和高阈值
    cv2.imshow('Edge Detection', edges)
    (cuntours,_)=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    bladder=[]
    for c in cuntours:
        for point in c:
            x, y = point[0]  
            if bx1 < x < bx2 and by1 < y < by2: 
                cv2.drawContours(img, [point], -1, (255, 0, 0), 2)
                if tuple(point[0]) not in bladder: 
                    bladder.append(tuple(point[0]))
    bladder = [list(t) for t in bladder]
    sorted_bladder = sorted(bladder, key=lambda point: point[0])
    for c in sorted_bladder:
        cv2.circle(img, tuple(c), 5, (0, 255, 0), -1)

    

# 显示分割结果
    cv2.imshow('Thresholded Image', thresholded)
    cv2.imshow('Filled Image', fill_img)
    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象并关闭窗口
cap.release()
cv2.destroyAllWindows() 