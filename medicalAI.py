
import cv2
import numpy as np

#基本參數
winit = True                                                      #初始判斷
init = True                                                       #初始判斷
Jflag = False
Wflag = False
test_suc = True
test_fail = True
tsflag = False
tsflag2 = False
rightflag = True
leftflag = True
find_mid = 1000
test_excer = 0
excercise = 0                                                      #運動次數
framecnt = 0                                                           #計算關鍵幀(建立初始距離)
gap_L = 0                                                         #左膀胱壁距
static_gap_L = 0                                                  #初始左膀胱壁距
shortest_gap_L = 999                                              #最短左膀胱壁距
longest_gap_L = 0                                                 #最長左膀胱壁距
gap_R = 0                                                         #右膀胱壁距
static_gap_R = 0                                                  #初始右膀胱壁距
shortest_gap_R = 999                                              #最短右膀胱壁距
longest_gap_R = 0                                                 #最長右膀胱壁距
lower = np.array([0,0,0])                                      #顏色範圍下限
upper = np.array([40,40,40])                                   #顏色範圍上限
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
    bott_checkpoint = []

    #驗證範圍框選
    for c in internal_contours:
        cv2.circle(img, (0, 0), 5, (0,0,255), -1)
        contour_points = []
        for point in c:
            x, y = point[0]  
            if 250 < x < 450 and 150 < y < 300:
                if 150 < y < 200:
                   top_checkpoint.append(point[0])
                if 250 < y <400:
                    bott_checkpoint.append(point[0])  
                cv2.drawContours(img, [point], -1, (255, 0, 0), 2)  

    #驗證點選取
    top_certify = [0,0]
    the_mid = 0
    mid_cnt = 0 
    midter = 0 
    got = False
    sorted_bott_checkpoint = sorted(bott_checkpoint, key=lambda point: point[0]) 
    top_certify = top_checkpoint[len(top_checkpoint)//2]
    the_mid = min(sorted_bott_checkpoint, key=lambda x: abs(x[0] - top_certify[0]))

    for mid in sorted_bott_checkpoint:
        if mid[0] < the_mid[0]:
            mid_cnt += 1
    bott_certify_left = sorted_bott_checkpoint[mid_cnt//2]
    if (len(sorted_bott_checkpoint) - mid_cnt) < (mid_cnt + 5):
        bott_certify_right = [0,0]
    bott_certify_right = sorted_bott_checkpoint[(len(sorted_bott_checkpoint)- mid_cnt)//2]
    print("len", len(sorted_bott_checkpoint))
    print("mid", mid_cnt)
    if framecnt == 0:
        init_topc = top_certify
        init_bottcl = bott_certify_left
        init_bottcr = bott_certify_right
    ct = 0
    blct = 0
    brct = 0
    cth = 0
    blcth = 0
    brcth = 0
    count = 0
    for i in top_checkpoint:
        
        ct += i[0]
        cth +=i[1]
    checsum = len(top_checkpoint)
    ct = ct//checsum
    for k in sorted_bott_checkpoint:
        if k[0] < ct:
            blct += k[0]
            count += 1
        else:
            brct += k[0]
        if k[0] < ct:
            blcth += k[1]
        else:
            brcth += k[1]   
    bchecsum = len(bott_checkpoint) - count
    
    if count ==0:
        blct = blct
        blcth = blcth
    else:
        blct = blct//count
        blcth = blcth//count
    if bchecsum != 0:
        brct = brct//bchecsum
        brcth = brcth//bchecsum
        rightflag = True
    else:
        brcth = 0
        brcth = 0
        rightflag = False
    cth = cth//checsum
    
    if framecnt == 0:
       Ttake1 = cth
       Ltake1 = blcth
       Rtake1 = brcth
       print("Ttake1",Ttake1)
       print("Ltake1",Ltake1)
       print("Rtake1",Rtake1)
    gbl = f'gapblcth:{blcth}'
    cv2.putText(img, gbl, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 2)
    gbr = f'gapbrcth:{brcth}'
    cv2.putText(img, gbr, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 2)
    gct = f'gapcth:{cth}'
    cv2.putText(img, gct, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 2)
    Ttake2 = cth
    Ltake2 = blcth
    Rtake2 = brcth
    if Ttake2 - Ttake1 > 10 or Ttake1 - Ttake2 > 10:
        Ttake1 = Ttake2
        Tt = f'Ttake1:{Ttake1}'
        cv2.putText(img, Tt, (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 2)
    if Ltake2 - Ltake1 > 3 or Ltake1 - Ltake2 > 3:
        Ltake1 = Ltake2
        Lt = f'Ltake1:{Ltake1}'
        cv2.putText(img, Lt, (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        init = True
    if Rtake2 - Rtake1 > 3 or Rtake1 - Rtake2 > 3:
        Rtake1 = Rtake2
        Rt = f'Rtake1:{Rtake1}'
        cv2.putText(img, Rt, (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        init = True
    else:
        init = False

    if framecnt == 0:
        init_testRgap = Rtake1 - Ttake1
        init_testLgap = Ltake1 - Ttake1

    testRgap = Rtake1 - Ttake1
    testLgap = Ltake1 - Ttake1
    TRG = f'TRG:{testRgap}'
    TLG = f'TLG:{testLgap}'
    ITRG = f'ITRG:{init_testRgap}'
    ITLG = f'ITLG:{init_testLgap}'
    cv2.putText(img, ITRG, (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, ITLG, (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, TRG, (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, TLG, (100, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    if max(init_testLgap - testLgap ,init_testRgap - testRgap ) > max(testLgap - init_testLgap ,testRgap - init_testRgap) and brcth != 0 and blcth != 0 :
        cv2.circle(img,(50,500), 10, (255,100,100), -1)
        if tsflag == False and init == True:
            test_excer += 1
            tsflag = True
            tsflag2 = True
    if (testLgap - init_testLgap > 7 or testRgap - init_testRgap > 7) or brcth == 0 :
        cv2.circle(img,(50 ,550), 10, (255,0,0), -1)
        if (tsflag == True and init == True) and test_excer > 0 :
            test_excer -= 1
            tsflag = False
    if init_testLgap - testLgap < 3 and testLgap - init_testLgap < 3 and init_testRgap - testRgap < 3 and testRgap - init_testRgap < 3:
        tsflag = False
        tsflag2 = False
        cv2.circle(img,(50 ,550), 10, (0,0,255), -1)

        


    #驗證點寫出
    for c in internal_contours:
        for point in c:
            x,y=point[0]
            if  x == top_certify[0] and y == top_certify[1]:
                cv2.circle(img,(x ,y), 5, (0,0,255), -1)
            if  x == bott_certify_left[0] and y == bott_certify_left[1]:
                cv2.circle(img,(x ,y), 5, (0,0,255), -1)
            if  x == bott_certify_right[0] and y == bott_certify_right[1]:
                cv2.circle(img,(x ,y), 5, (0,0,255), -1)
    
    #數據計算
    gap_L = bott_certify_left[1] - top_certify[1]
    gap_R = bott_certify_right[1] - top_certify[1]
    if framecnt == 0:
        static_gap_L = gap_L
        static_gap_R = gap_R
        
    if gap_L < shortest_gap_L:
        shortest_gap_L = gap_L
        
    if gap_R < shortest_gap_R:
        shortest_gap_R = gap_R
    if gap_L > longest_gap_L:
        longest_gap_L = gap_L
    if gap_R > longest_gap_R:
        longest_gap_R = gap_R

    #數據寫出
    Lap = f'gap_L:{gap_L}'
    Rap = f'gap_R:{gap_R}'
    sta_Lap = f'static_gap_L:{static_gap_L}'
    sta_Rap = f'static_gap_R:{static_gap_R}'
    sho_Lap = f'shortest_gap_L:{shortest_gap_L}'
    sho_Rap = f'shortest_gap_R:{shortest_gap_R}'
    lon_Lap = f'longest_gap_L:{longest_gap_L}'
    lon_Rap = f'longest_gap_R:{longest_gap_R}'
    cv2.putText(img, sta_Lap, (600, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
    cv2.putText(img, sta_Rap, (750, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
    cv2.putText(img, sho_Lap, (600, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
    cv2.putText(img, sho_Rap, (750, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
    cv2.putText(img, lon_Lap, (600, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
    cv2.putText(img, lon_Rap, (750, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
    cv2.putText(img, Rap, (750, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
    cv2.putText(img, Lap, (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
    cv2.circle(img,(ct ,cth), 5, (255,0,255), -1)
    cv2.circle(img,(blct ,blcth), 5, (255,0,255), -1)
    cv2.circle(img,(brct ,brcth), 5, (255,0,255), -1)
    if max(static_gap_L - shortest_gap_L , static_gap_R - shortest_gap_R) > max(longest_gap_L - static_gap_L , longest_gap_R - static_gap_R) and bott_certify_right[0] != 0 :
        succ = f'excerise : success'
        cv2.putText(img, succ, (600, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
        if init == True and Jflag == False and bott_certify_right[0] != 0:
            excercise += 1
            init = False
            Jflag = True
            cv2.circle(img,(100 ,400), 7, (255,255,0), -1)
    elif max(static_gap_L - shortest_gap_L , static_gap_R - shortest_gap_R) < max(longest_gap_L - static_gap_L , longest_gap_R - static_gap_R)  and max(longest_gap_L - static_gap_L , longest_gap_R - static_gap_R) >30: 
        succ = f'excerise : fail'
        cv2.putText(img, succ, (600, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
        if winit == True and excercise != 0 and Jflag == True:
            excercise -= 1
            winit = False
        
    else:
        succ = f'excerise : none'
        cv2.putText(img, succ, (600, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
        init = True
        Jflag = False
        winit = True
        
    

   #影像輸出
    cv2.imshow("outmask", img)

    framecnt += 1
    if cv2.waitKey(1) == ord('q'):
        break

#檢查數據用
print("static_gap_L",static_gap_L)
print("static_gap_R",static_gap_R)
print("shortest_gap_L",shortest_gap_L)
print("shortest_gap_R",shortest_gap_R)
print("longest_gap_L",longest_gap_L)
print("longest_gap_R",longest_gap_R)
print("static_gap_L - shortest_gap_L",static_gap_L - shortest_gap_L)
print("longest_gap_L - static_gap_L",longest_gap_L - static_gap_L)
print("static_gap_R - shortest_gap_R",static_gap_R - shortest_gap_R)
print("longest_gap_R - static_gap_R",longest_gap_R - static_gap_R)
print("test_excer",test_excer)


if max(static_gap_L - shortest_gap_L , static_gap_R - shortest_gap_R) > max(longest_gap_L - static_gap_L , longest_gap_R - static_gap_R):
    print("運動成功")
else:
    print("運動失敗")
print("運動次數",excercise)
cap.release()
#output_video.release()
cv2.destroyAllWindows()