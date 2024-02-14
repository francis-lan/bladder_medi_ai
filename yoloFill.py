import cv2
import numpy as np
from ultralytics import YOLO

movement = []
del_corrects = []
del_wrongs = []
del_list = []
correct_work_times = 0
wrong_work_times = 0
correct = False
wrong = False
adjust = 5
adj_list=[]
jump = False
mid_but_init = None
left_but_init = None
right_but_init = None

def pre_correct_count(movement):
    pcc = []
    for i in range(len(movement)-1, 0, -1):
        if movement[i] < 0:
            pcc.append(movement[i])
        elif movement[i] > 0:
            return pcc

def pre_wrong_count(movement):
    pwc = []
    for i in range(len(movement)-1, 0, -1):
        if movement[i] < 0:
            pwc.append(movement[i])
        elif movement[i] > 0:
            return pwc
        
#循環尋找最低邊界點
def FF_ver_cycle(img, seed_point_right,seed_point_left ,seed_y,threshold, y1,y2):
    min = [10000,0]
    for x in range(seed_point_left, seed_point_right):                                       #從左到右找最低點
        temp = FF_ver_down(img, [x,seed_y], threshold,y1,y2)
        if temp[1] > min[1]:
            min = temp
    if (min[0] >= seed_point_left and min[0] < seed_point_left + 10) or min[0] <= seed_point_right and min[0] > seed_point_right - 10:                                #如果符合條件代表膀胱底是近乎平滑的一元直線 無明顯膀胱底，用中央點來當最低點
        return FF_ver_down(img, [(seed_point_left + 50),seed_y], threshold,y1,y2)
    return min
#從seed開始往下找邊界
def FF_ver_down(img, seed_point, threshold,y1,y2):
    match_points=[]
    seed_color = (0, 0, 0)
    for y in range(seed_point[1]-20,y2):
        current_color = img[y, seed_point[0]] 
        color_diff = (abs(int(current_color[0]) - int(seed_color[0])), abs(int(current_color[1]) - int(seed_color[1])), abs(int(current_color[2]) - int(seed_color[2])))
        if all(diff <= threshold for diff in color_diff):
            match_points.append((seed_point[0], y))
            
        else:
            break
        
    if len(match_points) == 0:
        
        return seed_point
    else:
        return match_points[len(match_points)-1]
#從seed開始往上找邊界
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
        
    
    if len(match_points) == 0:
        return seed_point
    else:
        return match_points[len(match_points)-1]
#從seed開始往右找邊界
def FF_ver_right(img, seed_point, threshold, x1, x2):
    match_points=[]
    seed_color = (0, 0, 0)
    for x in range(seed_point[0], x2):
        current_color = img[int(seed_point[1]), x] 
        color_diff = (abs(int(current_color[0]) - int(seed_color[0])), abs(int(current_color[1]) - int(seed_color[1])), abs(int(current_color[2]) - int(seed_color[2])))
        if all(diff <= threshold for diff in color_diff):
            match_points.append((x, seed_point[1]))
        else:
            break
        
    
    if len(match_points) == 0:
        return seed_point
    else:
        return match_points[len(match_points)-1]
#從seed開始往左找邊界
def FF_ver_left(img, seed_point, threshold, x1, x2):
    match_points=[]
    seed_color = (0, 0, 0)
    for x in range(seed_point[0], x1, -1):
        current_color = img[int(seed_point[1]), x] 
        color_diff = (abs(int(current_color[0]) - int(seed_color[0])), abs(int(current_color[1]) - int(seed_color[1])), abs(int(current_color[2]) - int(seed_color[2])))
        if all(diff <= threshold for diff in color_diff):
            match_points.append((x, seed_point[1]))
        else:
            break
        
    
    if len(match_points) == 0:
        return seed_point
    else:
        return match_points[len(match_points)-1]
#在底部趨於一元直線時進行資料處裡
def FF_parr(img, seed_point,left_point, right_point, threshold, y1, y2):
    
    mid_but  = FF_ver_down(img, seed_point, threshold, y1, y2)
    left_but  = FF_ver_down(img, left_point, threshold, y1, y2)
    right_but  = FF_ver_down(img, right_point, threshold, y1, y2)
    left_top  = FF_ver_top(img, left_point, threshold, y1, y2)
    right_top  = FF_ver_top(img, right_point, threshold, y1, y2)
    mid_top  = FF_ver_top(img, seed_point, threshold, y1, y2)
    left_gap = left_but[1] - left_point[1]
    right_gap = right_but[1] - right_point[1]
    mid_gap = mid_but[1] - seed_point[1]
    cv2.line(img, (left_but[0], left_but[1]), (left_point[0], left_point[1]), (0, 0, 255), 2)
    cv2.line(img, (right_but[0], right_but[1]), (right_point[0], right_point[1]), (0, 0, 255), 2)
    cv2.line(img, (mid_but[0], mid_but[1]), (seed_point[0], seed_point[1]), (0, 0, 255), 2)

    return left_gap, right_gap, mid_gap
#在底部為凹口向上時進行資料處裡
def FF_trian(img, seed_point,mid_but,left_but,right_but, threshold, y1, y2,x1,x2):
    
    
    ML_gap = round(np.sqrt((mid_but[0] - left_but[0]) * (mid_but[0] - left_but[0]) + (left_but[1] - mid_but[1]) * (left_but[1] - mid_but[1])),3)
    MR_gap = round(np.sqrt((mid_but[0] - right_but[0]) * (mid_but[0] - right_but[0]) + (right_but[1] - mid_but[1]) * (right_but[1] - mid_but[1])),3)
    LR_gap = round(np.sqrt((left_but[0] - right_but[0]) * (left_but[0] - right_but[0]) + (left_but[1] - right_but[1]) * (left_but[1] - right_but[1])),3)
    delta_LM = round((left_but[1] - mid_but[1])/(left_but[0] - mid_but[0]),3)
    delta_MR =round((right_but[1] - mid_but[1])/(right_but[0] - mid_but[0]),3)
    delta_LR = round((left_but[1] - right_but[1])/(left_but[0] - right_but[0]),3)
    tan_but = round((delta_LM - delta_MR) / (1 - delta_LM * delta_MR),3)
    mid_gap = round(mid_but[1] - seed_point[1],3)
    c = left_but[1] - delta_LR * left_but[0]
    mid_on_line = delta_LR * mid_but[0] + c
    mid_train_gap = mid_but[1] - mid_on_line


    cv2.line(img, (int(left_but[0]), int(left_but[1])), (int(right_but[0]), int(right_but[1])), (0, 0, 255), 2)
    cv2.line(img, (int(left_but[0]), int(left_but[1])), (int(mid_but[0]), int(mid_but[1])), (0, 0, 255), 2)
    cv2.line(img, (int(right_but[0]), int(right_but[1])), (int(mid_but[0]), int(mid_but[1])), (0, 0, 255), 2)

    return ML_gap, LR_gap,MR_gap,delta_LM,delta_MR,tan_but,mid_gap

#判斷前一個值
def non_zero_pre(move):
    for i in range(len(move)-1,0,-1):
        if move[i] < 0:
            return True
        elif move[i] > 0:
            return False

#總體運動判斷
def excer_check(del_LM, del_RM, mid_but, img):
    global correct_work_times, wrong_work_times, del_corrects, del_wrongs, del_list,correct,adjust,adj_list,jump,wrong,movement
    LRM = [del_LM, del_RM, mid_but[1]]
    del_list.append(LRM)
    temp = []

    if len(del_list) <= 2:
        return
    else: 
        #當中央點上移
        if del_list[len(del_list)-1][2] < del_list[len(del_list)-2][2] :
            if non_zero_pre(movement) == False:
                if len(del_corrects) == 0:
                    del_wrongs.clear()
                elif len(del_corrects) > 0 and len(del_wrongs) > 0:  
                    judge = min(del_corrects[0][2],del_wrongs[len(del_wrongs)-1][2])
                    if judge - del_corrects[len(del_corrects)-1][2] > 4:
                        correct_work_times += 1
                        del_corrects.clear()
                        del_wrongs.clear()
                    elif judge == del_corrects[len(del_corrects)-1][2]:
                        del_corrects.clear()
                        del_wrongs.clear()
                else:
                    if len(del_corrects) > 0:
                        if del_corrects[len(del_corrects)-1][2] < del_list[len(del_list)-1][2]:
                            if del_list[len(del_list)-1][2] - del_corrects[len(del_corrects)-1][2] < 1:
                                temp = pre_wrong_count(movement)
                                for i in temp:
                                    del_wrongs.pop(i)
                        else:
                            del_corrects.append(del_list[len(del_list)-1])
                            movement.append(-len(del_corrects))
                    else:
                        del_corrects.append(del_list[len(del_list)-1])
                        movement.append(-len(del_corrects))
            else:        
                del_corrects.append(del_list[len(del_list)-1])
                movement.append(-len(del_corrects))
        elif del_list[len(del_list)-1][2] > del_list[len(del_list)-2][2]  :
            if non_zero_pre(movement) == True:
                if len(del_wrongs) > 0:
                    if del_list[len(del_list)-1][2] < del_wrongs[len(del_wrongs)-1][2]:
                        if del_wrongs[len(del_wrongs)-1][2] - del_list[len(del_list)-1][2] < 1:
                            temp = pre_correct_count(movement)
                            for i in temp:
                                del_corrects.pop(i)
                    else:
                        del_wrongs.append(del_list[len(del_list)-1])
                        movement.append(len(del_wrongs))
                else:
                    del_wrongs.append(del_list[len(del_list)-1])
                    movement.append(len(del_wrongs))
            else:      
                del_wrongs.append(del_list[len(del_list)-1])
                movement.append(len(del_wrongs))
        elif del_list[len(del_list)-1][2] <= del_list[len(del_list)-2][2] + 0.3 and del_list[len(del_list)-1][2] >= del_list[len(del_list)-2][2] - 0.3:
            movement.append(0)
            cv2.circle(img, (100,500), 5, (255, 255, 0), -1)
            
    
    
    
#軌跡跟蹤
def draw_flow(img, p0, p1):
    for i, (new, old) in enumerate(zip(p1, p0)):
        a, b = new.ravel()
        c, d = old.ravel()
        img = cv2.line(img, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)
        img = cv2.circle(img, (int(a), int(b)), 5, (0, 0, 255), -1)
    return img    
    
    
#AI本體 
def main(): 
    global mid_but_init, left_but_init,right_but_init 
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    threshold = 90
    traingle = False
    parr = False
    model = YOLO("./runs/detect/train7/weights/best.pt")     #找膀胱位置用模型
    #cap = cv2.VideoCapture('./source_pack/1701332680.mp4')
    #cap = cv2.VideoCapture('./source_pack/kegal_2.mp4')
    #cap = cv2.VideoCapture('./source_pack/kegal_1.mp4')
    cap = cv2.VideoCapture('./source_pack/kegal_keep1.mp4')
    #cap = cv2.VideoCapture('./source_pack/kegal_keep2.mp4')
    #cap = cv2.VideoCapture("./source_pack/1701334235.mp4")
    #cap = cv2.VideoCapture("./source_pack/1701332749.mp4")
    #cap = cv2.VideoCapture("./source_pack/1701332680.mp4")
    #cap = cv2.VideoCapture('./source_pack/TaUS_K1(kwT).mp4')
    #cap = cv2.VideoCapture('./source_pack/TaUS_V(Wrong).mp4')
    #cap = cv2.VideoCapture(0)                                          #開鏡頭用的
    frame_rate = int(cap.get(5))                                      #影片幀率
    x1, y1, x2, y2 = 100, 0, 700, 600                                 #剪裁範圍
    gap = []
    point = []
    huge_gap = []

    seed_x = 0
    seed_y = 0

    


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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
        #results = model(img)                      #yolo直接預測
        setxyxy = model.predict(img)
        
        #把邊界點抓出來
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
         #設定種子
        if((bx1 + bx2)//2 - seed_x > 5 and (by1 + by2)//2 - seed_y >5):
            seed_point = ((bx1 + bx2)//2, (by1+by2)//2)
            left_point = (((bx1+bx2)//2 + bx1)//2, (by1+by2)//2)
            right_point = (((bx1+bx2)//2 + bx2)//2, (by1+by2)//2)
            seed_x = (bx1 + bx2)//2
            seed_y = (by1 + by2)//2
        #光流追蹤初始點
        if mid_but_init is None:
            mid_but_init = FF_ver_cycle(img, (int(seed_point[0]) + 50),(int(seed_point[0]) - 50), seed_point[1],threshold, y1, y2)
            left_but_init = FF_ver_down(img, left_point, threshold, y1, y2)
            right_but_init = FF_ver_down(img, right_point, threshold, y1, y2)
            verify_point = [mid_but_init[0], mid_but_init[1] - 80]
            core_previous = ((int(mid_but_init[0]) + int(right_but_init[0]) + int(left_but_init[0]) // 3), (int(mid_but_init[1]) + int(right_but_init[1]) + int(left_but_init[1]) // 3))
            mid_previous = (int(mid_but_init[0]), int(mid_but_init[1]))

        good_new = None
        good_old = None
        if 'old_gray' not in locals():
            old_gray = gray
            p0 = np.array([mid_but_init, left_but_init, right_but_init], dtype=np.float32)
            continue

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
        if len(st.shape) == 1:
            st = st.reshape(-1, 1)

        good_new = np.array([])
        good_old = np.array([])
        #光流追蹤點位置
        if st.shape[0] > 0:
            good_old = p0[st[:, 0] == 1]
            good_new = p1[st[:, 0] == 1]

            mid_but, left_but, right_but = good_new.reshape(-1, 2)
            mid_current = (int(mid_but[0]), int(mid_but[1]))
            core_current = ((int(mid_but[0]) + int(right_but[0]) + int(left_but[0]) // 3), (int(mid_but[1]) + int(right_but[1]) + int(left_but[1]) // 3))
            movement_vector_x = (mid_current[0] - mid_previous[0], mid_current[1] - mid_previous[1])
            movement_vector_y = (core_current[0] - core_previous[0], core_current[1] - core_previous[1])
            verify_point[0] += movement_vector_x[0]
            if  abs(movement_vector_y[0]) >  abs(movement_vector_y[1]) + 5:
                verify_point[1] += movement_vector_y[1]

        img = cv2.circle(img, (int(mid_but[0]), int(mid_but[1])), 5, (0, 255, 0), -1)
        img = cv2.circle(img, (int(left_but[0]), int(left_but[1])), 5, (0, 255, 0), -1)
        img = cv2.circle(img, (int(right_but[0]), int(right_but[1])), 5, (0, 255, 0), -1)
        img = cv2.circle(img, (int(verify_point[0]), int(verify_point[1])), 5, (255, 0, 0), -1)
        img = draw_flow(img, good_old, good_new)
        
        core_previous = core_current
        mid_previous = mid_current




        #left_but  = FF_ver_down(img, left_point, threshold, y1, y2)
        #right_but  = FF_ver_down(img, right_point, threshold, y1, y2)
        #mid_but  = FF_ver_down(img, seed_point, threshold, y1, y2)
        gg=[]
        deal = []
        #開始判讀
        if(abs(left_but[1] - mid_but[1]) < 10) and (abs(right_but[1] - mid_but[1]) < 10) and traingle == False:
            left_gap, right_gap, mid_gap = FF_parr(img, seed_point,left_point, right_point, threshold, y1, y2)
            
            print("平行處理")
            parr = True
        elif parr == False:
            ML_gap, LR_gap,MR_gap, del_LM, del_RM ,tan_But,mid_gap= FF_trian(img, verify_point, mid_but, left_but, right_but, threshold, y1, y2,x1,x2)
            excer_check(del_LM, del_RM, mid_but, img)
            print("三角處理")
            gg = [ML_gap, mid_gap,MR_gap,LR_gap]
            deal = [del_LM, del_RM]

            gap.append(gg)
            point.append(deal)
            huge_gap.append(tan_But)
            traingle = True
            

            


   
    
        cv2.imshow("img", img)
        old_gray = gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    print("gap = ", gap)
    print("point = ", point)
    print("huge_gap = ", huge_gap)
    print("correct_work_times = ", correct_work_times)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()