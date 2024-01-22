import cv2
import numpy as np
import matplotlib.pyplot as plt



  
def topcurve(arr = []):
     
    points = np.squeeze(arr)
    # 多项式拟合
    degree = 8  # 选择多项式的阶数
    coefficients = np.polyfit(points[:, 0], points[:, 1], degree)
    poly = np.poly1d(coefficients)
    #result = poly(100)
    # 生成拟合曲线上的点
   # x_vals = np.linspace(min(points[:, 0]), max(points[:, 0]), 100)
   # y_vals = poly(x_vals)
    # 可视化拟合曲线和原始轮廓点
   # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
   # ax.plot(x_vals, y_vals, label='Fitted Curve', color='red')
   # ax.scatter(points[:, 0], points[:, 1], label='Contour Points', color='blue')
   # ax.set_xlabel('X-axis')
   # ax.set_ylabel('Y-axis')
   # ax.set_title('Fitted Curve for Contour Points')
    #ax.legend()
    #plt.show()
    return poly

def rightcurve(arr = []):
    points = np.squeeze(arr)
    
    degree = 8  
    coefficients = np.polyfit(points[:, 0], points[:, 1], degree)
    poly = np.poly1d(coefficients)
    #x_vals = np.linspace(min(points[:, 0]), max(points[:, 0]), 100)
    #y_vals = poly(x_vals)
   
   # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    #ax.plot(x_vals, y_vals, label='Fitted Curve', color='red')
    #ax.scatter(points[:, 0], points[:, 1], label='right Points', color='blue')
    #ax.set_xlabel('X-axis')
    #ax.set_ylabel('Y-axis')
   # ax.set_title('Fitted Curve for Contour Points')
   # ax.legend()
    #plt.show()
    return poly
    

def leftcurve(arr = []):
     
    points = np.squeeze(arr)
    # 多项式拟合
    degree = 8  # 选择多项式的阶数
    coefficients = np.polyfit(points[:, 0], points[:, 1], degree)
    poly = np.poly1d(coefficients)
    
    # 生成拟合曲线上的点
    #x_vals = np.linspace(min(points[:, 0]), max(points[:, 0]), 100)
   # y_vals = poly(x_vals)
    # 可视化拟合曲线和原始轮廓点
   # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
   # ax.plot(x_vals, y_vals, label='Fitted Curve', color='red')
   # ax.scatter(points[:, 0], points[:, 1], label='left Points', color='blue')
   # ax.set_xlabel('X-axis')
   # ax.set_ylabel('Y-axis')
   # ax.set_title('Fitted Curve for Contour Points')
   # ax.legend()
    #plt.show()
    return poly



#基本參數

framecnt = 0  
top_poly = []
right_poly = []
left_poly = []                                          
lower = np.array([60,60,60])                                      #顏色範圍下限
upper = np.array([255,255,255])                                   #顏色範圍上限
cap = cv2.VideoCapture('D:/User-Data/Downloads/kegal_keep1.mp4')
#cap = cv2.VideoCapture('D:/User-Data/Downloads/kegal_keep2.mp4')
#cap = cv2.VideoCapture('D:/User-Data/Downloads/TaUS_K1(kwT).mp4')
#cap = cv2.VideoCapture('D:/User-Data/Downloads/TaUS_V(Wrong).mp4')
#cap = cv2.VideoCapture(0)

#剪裁影片用參數
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
    img  = black_background                                       #影像更新

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                  #轉成二元
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)                 #高斯模糊
    binaryIMG = cv2.Canny(blurred, 20, 160)                       #邊緣偵測
    mask = cv2.inRange(img, lower, upper)                         # 取得顏色範圍
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))  # 設定膨脹與侵蝕的參數
    mask = cv2.dilate(mask, kernel)                               # 膨脹影像，消除雜訊
    mask = cv2.erode(mask, kernel)                                # 縮小影像，還原大小
    
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
    top_check_nupy = []
    bott_checkpoint = []
    bott_check_nupy = []
    #驗證範圍框選
    for c in internal_contours:
        cv2.circle(img, (0, 0), 5, (0,0,255), -1)
        contour_points = []
        for point in c:
            x, y = point[0]  
            if 250 < x < 450 and 150 < y < 300:
                if 150 < y < 200:
                   top_checkpoint.append(point[0])
                   top_check_nupy.append([x, y])
                if 250 < y <400:
                    bott_checkpoint.append(point[0])
                    bott_check_nupy.append([x, y])
                cv2.drawContours(img, [point], -1, (255, 0, 0), 2)  
    

    sorted_bott_checkpoint = sorted(bott_checkpoint, key=lambda point: point[0]) 
    sorted_bott_check_nupy = sorted(bott_check_nupy, key=lambda point: point[0])
    right_nupy = []
    left_nupy = []
    ct = 0
    mid_cnt = 0
    for i in top_checkpoint:
        
        ct += i[0]
    checsum = len(top_checkpoint)
    ct = ct//checsum
    for i in sorted_bott_checkpoint:
        if i[0] <ct:
            mid_cnt += 1
    for i in range(0,len(sorted_bott_check_nupy)):
        if i < mid_cnt:
            left_nupy.append(sorted_bott_check_nupy[i])
        else:
            right_nupy.append(sorted_bott_check_nupy[i])
    if framecnt == 0:
        top_poly = topcurve(top_check_nupy)
        right_poly = rightcurve(right_nupy)
        left_poly = leftcurve(left_nupy)
        

    
    framecnt += 1
    if cv2.waitKey(1) == ord('q'):
        break


print("top poly :","/n",top_poly,"/n")
print("right poly :","/n",right_poly,"/n")
print("left poly :","/n",left_poly,"/n")


cap.release()
#output_video.release()
cv2.destroyAllWindows()