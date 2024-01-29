import cv2
import numpy as np
from ultralytics import YOLO

del_corrects = []
del_wrongs = []
del_list = []
correct_work_times = 0
wrong_work_times = 0
correct = False
adjust = 5
adj_list=[]
jump = False
mid_but_init = None
left_but_init = None
right_but_init = None

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

def FF_trian(img, seed_point,mid_but,left_but,right_but, threshold, y1, y2,x1,x2):
    
    
    ML_gap = (mid_but[0] - left_but[0]) * (mid_but[0] - left_but[0]) + (left_but[1] - mid_but[1]) * (left_but[1] - mid_but[1])
    MR_gap = (mid_but[0] - right_but[0]) * (mid_but[0] - right_but[0]) + (right_but[1] - mid_but[1]) * (right_but[1] - mid_but[1])
    LR_gap = (left_but[0] - right_but[0]) * (left_but[0] - right_but[0]) + (left_but[1] - right_but[1]) * (left_but[1] - right_but[1])
    delta_LM = round((left_but[1] - mid_but[1])/(left_but[0] - mid_but[0]),3)
    delta_MR =round((right_but[1] - mid_but[1])/(right_but[0] - mid_but[0]),3)
    tan_but = round((delta_LM - delta_MR) / (1 - delta_LM * delta_MR),3)


    cv2.line(img, (int(left_but[0]), int(left_but[1])), (int(right_but[0]), int(right_but[1])), (0, 0, 255), 2)
    cv2.line(img, (int(left_but[0]), int(left_but[1])), (int(mid_but[0]), int(mid_but[1])), (0, 0, 255), 2)
    cv2.line(img, (int(right_but[0]), int(right_but[1])), (int(mid_but[0]), int(mid_but[1])), (0, 0, 255), 2)

    return ML_gap, LR_gap,MR_gap,delta_LM,delta_MR,tan_but
    
def excer_check(del_LM, del_RM, mid_but, img):
    global correct_work_times, wrong_work_times, del_corrects, del_wrongs, del_list,correct,adjust,adj_list,jump
    LRM = [del_LM, del_RM, mid_but[1]]
    del_list.append(LRM)

    if len(del_list) == 1:
        return
    else: 
        if del_list[0][2] + adjust < del_list[len(del_list) - 1][2]:
            del_wrongs.append(del_list[len(del_list) - 1])
            del_corrects.clear()
            #gap = del_list[len(del_list) - 1][2] - del_list[0][2]
            #jump = True
            #adj_list.append(gap)
            #if len(adj_list) > 20:
                #adjust = sum(adj_list)/len(adj_list)
                #adj_list.clear()
            correct = False
            cv2.circle(img, (200,500), 5, (0, 255, 0), -1)
        else:
            #if jump == True and del_list[0][2] == del_list[len(del_list) - 1]:
                #adj_list.clear()
            #jump = False
            if (del_list[len(del_list) - 1][0] == del_list[0][0] and del_list[0][1] == del_list[len(del_list) - 1][1]) or del_list[len(del_list) - 1][2] == del_list[0][2]:
                del_corrects.clear()
                cv2.circle(img, (200,500), 5, (0, 255, 0), -1)
            elif del_list[len(del_list) - 1][0] > del_list[len(del_list) - 2][0]:
                del_corrects.append(del_list[len(del_list) - 1])
                del_wrongs.clear()
                cv2.circle(img, (200,500), 5, (255, 0, 0), -1)
            elif del_list[len(del_list) - 1][0] < del_list[len(del_list) - 2][0]:
                if del_list[len(del_list) - 1][1] <  del_list[len(del_list) - 2][1] or del_list[len(del_list) - 1][1] > del_list[len(del_list) - 2][1]:
                    del_corrects.append(del_list[len(del_list) - 1])
                    del_wrongs.clear()
                    cv2.circle(img, (200,500), 5, (255, 0, 0), -1)
            else:
                if del_list[len(del_list) - 1][1] < del_list[len(del_list) - 2][1]:
                    del_corrects.append(del_list[len(del_list) - 1])
                    del_wrongs.clear()
                    cv2.circle(img, (200,500), 5, (255, 0, 0), -1)
                elif del_list[len(del_list) - 1][1] == del_list[len(del_list) - 2][1]:
                    if del_list[len(del_list) - 1][2] < del_list[len(del_list) - 2][2]:
                        del_corrects.append(del_list[len(del_list) - 1])
                        del_wrongs.clear()
                        cv2.circle(img, (200,500), 5, (255, 0, 0), -1)
                
    
    if len(del_corrects) == 8 : 
        correct_work_times += 1
        cv2.circle(img, (100,500), 5, (0, 0, 255), -1)
            
    
    elif len(del_wrongs) == 8 and del_wrongs[7][0] > del_list[0][0]:
        wrong_work_times += 1

def draw_flow(img, p0, p1):
    for i, (new, old) in enumerate(zip(p1, p0)):
        a, b = new.ravel()
        c, d = old.ravel()
        img = cv2.line(img, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)
        img = cv2.circle(img, (int(a), int(b)), 5, (0, 0, 255), -1)
    return img    
    
    
    
def main(): 
    global mid_but_init, left_but_init,right_but_init 
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    threshold = 90
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

        if((bx1 + bx2)//2 - seed_x > 5 and (by1 + by2)//2 - seed_y >5):
            seed_point = ((bx1 + bx2)//2, (by1+by2)//2)
            left_point = (((bx1+bx2)//2 + bx1)//2, (by1+by2)//2)
            right_point = (((bx1+bx2)//2 + bx2)//2, (by1+by2)//2)
            seed_x = (bx1 + bx2)//2
            seed_y = (by1 + by2)//2
        
        if mid_but_init is None:
            mid_but_init = FF_ver_down(img, seed_point, threshold, y1, y2)
            left_but_init = FF_ver_down(img, left_point, threshold, y1, y2)
            right_but_init = FF_ver_down(img, right_point, threshold, y1, y2)

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

        if st.shape[0] > 0:
            good_old = p0[st[:, 0] == 1]
            good_new = p1[st[:, 0] == 1]

            mid_but, left_but, right_but = good_new.reshape(-1, 2)

        img = cv2.circle(img, (int(mid_but[0]), int(mid_but[1])), 5, (0, 255, 0), -1)
        img = cv2.circle(img, (int(left_but[0]), int(left_but[1])), 5, (0, 255, 0), -1)
        img = cv2.circle(img, (int(right_but[0]), int(right_but[1])), 5, (0, 255, 0), -1)
        img = draw_flow(img, good_old, good_new)




        #left_but  = FF_ver_down(img, left_point, threshold, y1, y2)
        #right_but  = FF_ver_down(img, right_point, threshold, y1, y2)
        #mid_but  = FF_ver_down(img, seed_point, threshold, y1, y2)
        gg=[]
        deal = []

        if(abs(left_but[1] - mid_but[1]) < 10) and (abs(right_but[1] - mid_but[1]) < 10):
            left_gap, right_gap, mid_gap = FF_parr(img, seed_point,left_point, right_point, threshold, y1, y2)
            
            print("平行處理")
        else:
            ML_gap, LR_gap,MR_gap, del_LM, del_RM ,tan_But= FF_trian(img, seed_point, mid_but, left_but, right_but, threshold, y1, y2,x1,x2)
            excer_check(del_LM, del_RM, mid_but, img)
            print("三角處理")
            gg = [ML_gap, LR_gap,MR_gap]
            deal = [del_LM, del_RM]

            gap.append(gg)
            point.append(deal)
            huge_gap.append(tan_But)
            

            


   
    
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