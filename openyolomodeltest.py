import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("C:/vscode.ai/runs/detect/train7/weights/best.pt")
#cap = cv2.VideoCapture('D:/User-Data/Downloads/kegal_keep1.mp4')
#cap = cv2.VideoCapture('D:/User-Data/Downloads/kegal_keep2.mp4')
#cap = cv2.VideoCapture("D:/User-Data/Downloads/1701334235.mp4")
#cap = cv2.VideoCapture("D:/User-Data/Downloads/1701332749.mp4")
#cap = cv2.VideoCapture("D:/User-Data/Downloads/1701332680.mp4")
#cap = cv2.VideoCapture('D:/User-Data/Downloads/TaUS_K1(kwT).mp4')
cap = cv2.VideoCapture('D:/User-Data/Downloads/TaUS_V(Wrong).mp4')
#cap = cv2.VideoCapture(0)
frame_rate = int(cap.get(5))                                      #影片幀率
x1, y1, x2, y2 = 100, 0, 700, 600                                 #剪裁範圍
lower = np.array([65,65,65])                                     
upper = np.array([255,255,255])
lower2 = np.array([0,0,0])
upper2 = np.array([35,35,35]) 

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
    by2 = int(b[3]) - 20
    cv2.circle(img, (bx1, by1), 5, (0, 0, 255), -1)
    cv2.circle(img, (bx2, by2), 5, (0, 0, 255), -1)
    cv2.circle(img, (bx1, by2), 5, (0, 0, 255), -1)
    cv2.circle(img, (bx2, by1), 5, (0, 0, 255), -1)
    #annotated_frame = results[0].plot()                     #將預測後的結果直接畫上視窗
        # Display the annotated frame
    #cv2.imshow("YOLOv8 Inference", annotated_frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                  #轉成二元
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)                 #高斯模糊
    binaryIMG = cv2.Canny(blurred, 20, 160)                       #邊緣偵測
    mask = cv2.inRange(img, lower2, upper2)                         
    mask2 = cv2.inRange(img, lower, upper)                        #過濾顏色
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))  # 設定膨脹與侵蝕的參數
    mask = cv2.dilate(mask, kernel)                               # 膨脹影像，消除雜訊
    mask = cv2.erode(mask, kernel)                                # 縮小影像，還原大小
    mask2 = cv2.dilate(mask2, kernel)                               # 膨脹影像，消除雜訊
    mask2 = cv2.erode(mask2, kernel)                                # 縮小影像，還原大小
    
    #抓出所有輪廓
    (cnts, _) = cv2.findContours(binaryIMG.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clone = img.copy()

    #更新後的影像輸出
    cv2.imshow('Cropped Video', black_background)

    #繪製基礎輪廓
    (contours,_) = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    internal_contours = []
    for contour in contours:
        if not cv2.contourArea(contour):
            continue  
        if cv2.contourArea(contour) > 0:  
            internal_contours.append(contour)

    cv2.drawContours(img, internal_contours, -1, (0, 255, 0), 2)
    #驗證點陣列
    top_checkpoint = []
    bott_checkpoint = []

    #驗證範圍框選
    #for c in internal_contours:
        #cv2.circle(img, (0, 0), 5, (0,0,255), -1)
        #contour_points = []
        #for point in c:
            #x, y = point[0]  
            #if bx1 < x < bx2 and by1 < y < by2: 
               # cv2.drawContours(img, [point], -1, (255, 0, 0), 2)  

    (contours,_) = cv2.findContours(mask2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    internal_contours2 = []
    for contour in contours:
        if not cv2.contourArea(contour):
            continue  
        if cv2.contourArea(contour) > 0:  
            internal_contours2.append(contour)

    #cv2.drawContours(img, internal_contours2, -1, (0, 255, 0), 2)
    #驗證點陣列
    top_checkpoint = []
    bott_checkpoint = []

    #驗證範圍框選
    #for c in internal_contours2:
        #cv2.circle(img, (0, 0), 5, (0,0,255), -1)
        #contour_points = []
        #for point in c:
           # x, y = point[0]  
            #if bx1 < x < bx2 and by1 < y < by2: 
                #cv2.drawContours(img, [point], -1, (0, 0, 255), 2)  
    compare = []
    for c in internal_contours:
        for j in internal_contours2:
            if c.all() == j.all():
                compare.append(c)
    
    bladder = []
    for c in internal_contours:
        for point in c:
            x, y = point[0]  
            if bx1 < x < bx2 and by1 < y < by2: 
                cv2.drawContours(img, [point], -1, (255, 0, 0), 2)
                if tuple(point[0]) not in bladder: 
                    bladder.append(tuple(point[0]))
    bladder = [list(t) for t in bladder]
    sorted_bladder = sorted(bladder, key=lambda point: point[0])
   
    
    bladder_set = []
    tempsave = []
    for c in sorted_bladder:
        flag = False
        temp = []
        cnnt = 0
        for j in sorted_bladder:
            
            if c[0] == j[0]:
                if cnnt == 0:
                    temp.append(j[0])
                    cnnt += 1
                temp.append(j[1])
        for k in tempsave:
            if k == c[0]:
                flag = True
                break 
        if flag == False:
            bladder_set.append(temp)
            tempsave.append(c[0])
    bladder_set_final = []
    for c in bladder_set:
        if len(c) == 3:
            bladder_set_final.append(c)

    for c in bladder_set_final:
        if c[0] > (bx1 + bx2)//3 and c[0] < 2*(bx1 + bx2)//3 and abs(c[1] - c[2]) > 10 :
            cv2.circle(img, (c[0], c[1]), 5, (0, 0, 255), -1)
            cv2.circle(img, (c[0], c[2]), 5, (0, 0, 255), -1)
             


    cv2.imshow("img",img)
    

        # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    

print(bladder_set_final)
cap.release()
cv2.destroyAllWindows()