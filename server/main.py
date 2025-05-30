import cv2
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import norm
import argparse
import math

IMG_PATH = "source_img"
IMG_RESULT_PATH = "result_img"

def gaussian(x, sigma):
    return np.exp(- (x ** 2) / (2 * sigma ** 2))
@njit
def compute_result(h360, s, mask, template, new_h360, degree_diff):
    h, w = h360.shape
    m = template[1]
    a = template[2]
    print("m: ", m)
    arc_width = np.zeros((h, w), dtype=np.float32)
    center = np.zeros((h, w), dtype=np.float32)
    direction = np.zeros((h, w), dtype=np.float32)
    mean = 0
    if len(m) == 2:
        m0 = m[0]
        m1 = m[1]
        for i in range(h):
            for j in range(w):
                if mask[i, j] == 0:
                    p = h360[i, j]
                    l = (m0 + a) % 360
                    r = (m0 + m1 + a) % 360
                    values = [l, r]
                    values_list = [abs(p-values[0]), abs(p-values[1]), abs(p-values[0]+360), abs(p-values[1]+360)]
                    min_value = min(values_list)
                    min_index = values_list.index(min_value)
                    if r < l:  # 跨 0 度
                        if p >= l or p <= r:
                            arc_width[i, j] = m[1]
                            center[i, j] = (m0 + a + (m1/2)) % 360
                            degree_diff[i, j] = min_value
                            continue
                    elif l <= p <= r:
                        arc_width[i, j] = m[1]
                        center[i, j] = (m0 + a + (m1/2)) % 360
                        degree_diff[i, j] = min_value
                        continue
                    # 不在區間內
                    if(min_index % 2 == 0):
                        new_h360[i, j] = l
                    else:
                        new_h360[i, j] = r
                    arc_width[i, j] = m[1]
                    degree_diff[i, j] = min_value
                    center[i, j] = (m0 + a + (m1/2)) % 360
                    if(p - new_h360[i, j] >= -180 and p - new_h360[i, j] <= 180):
                        if(p - new_h360[i, j] >= 0):
                            direction[i, j] = 1
                        else:
                            direction[i, j] = -1
                    else:
                        if(p - new_h360[i, j] < -180):
                            direction[i, j] = 1
                        else:
                            direction[i, j] = -1
    elif len(m) == 4:
        m0, m1, m2, m3 = m
        for i in range(h):
            for j in range(w):
                if mask[i, j] == 0:
                    p = h360[i, j]
                    l1 = (m0 + a) % 360
                    r1 = (m0 + m1 + a) % 360
                    l2 = (m2 + a) % 360
                    r2 = (m2 + m3 + a) % 360

                    in1 = False
                    in2 = False
                    values = [l1, r1, l2, r2]
                    values_list = [abs(p-values[0]), abs(p-values[1]), abs(p-values[2]), abs(p-values[3]), abs(p-values[0]+360), abs(p-values[1]+360), abs(p-values[2]+360), abs(p-values[3]+360)]
                    min_value = min(values_list)
                    min_index = values_list.index(min_value)
                    if r1 < l1:
                        in1 = (p >= l1 or p <= r1)
                    else:
                        in1 = (l1 <= p <= r1)

                    if r2 < l2:
                        in2 = (p >= l2 or p <= r2)
                    else:
                        in2 = (l2 <= p <= r2)

                    if in1:
                        arc_width[i, j] = m[1]
                        center[i, j] = (m0 + a + (m1/2)) % 36
                        degree_diff[i, j] = min_value
                        if(p - new_h360[i, j] >= -180 and p - new_h360[i, j] <= 180):
                            if(p - new_h360[i, j] >= 0):
                                direction[i, j] = 1
                            else:
                                direction[i, j] = -1
                        else:
                            if(p - new_h360[i, j] < -180):
                                direction[i, j] = 1
                            else:
                                direction[i, j] = -1
                        continue
                    if in2:
                        arc_width[i, j] = m[3]
                        center[i, j] = (m2 + a + (m3/2)) % 360
                        degree_diff[i, j] = min_value
                        if(p - new_h360[i, j] >= -180 and p - new_h360[i, j] <= 180):
                            if(p - new_h360[i, j] >= 0):
                                direction[i, j] = 1
                            else:
                                direction[i, j] = -1
                        else:
                            if(p - new_h360[i, j] < -180):
                                direction[i, j] = 1
                            else:
                                direction[i, j] = -1
                        continue
                    
                    if (min_index % 4 == 0):
                        new_h360[i, j] = l1
                        arc_width[i, j] = m[1]
                        center[i, j] = (m0 + a + (m1/2)) % 360
                    elif (min_index % 4 == 1):
                        new_h360[i, j] = r1
                        arc_width[i, j] = m[1]
                        center[i, j] = (m0 + a + (m1/2)) % 360
                    elif (min_index % 4 == 2):
                        new_h360[i, j] = l2
                        arc_width[i, j] = m[3]
                        center[i, j] = (m2 + a + (m3/2)) % 360
                    else:
                        new_h360[i, j] = r2
                        arc_width[i, j] = m[3]
                        degree_diff[i, j] = min_value
                        center[i, j] = (m2 + a + (m3/2)) % 360
                    
                    if(p - new_h360[i, j] >= -180 and p - new_h360[i, j] <= 180):
                        if(p - new_h360[i, j] >= 0):
                            direction[i, j] = 1
                        else:
                            direction[i, j] = -1
                    else:
                        if(p - new_h360[i, j] < -180):
                            direction[i, j] = 1
                        else:
                            direction[i, j] = -1
    
            
    return  center, arc_width, degree_diff, direction

@njit
def compute_loss(m, a, h360, s, mask):
    radius = 1
    loss = 0.0
    h, w = h360.shape
    if len(m) == 2:
        m0 = m[0]
        m1 = m[1]
        for i in range(h):
            for j in range(w):
                if mask[i, j] == 1:
                    p = h360[i, j]
                    l = (m0 + a) % 360
                    r = (m0 + m1 + a) % 360
                    if r < l:  # 跨 0 度
                        if p >= l or p <= r:
                            continue
                    elif l <= p <= r:
                        continue
                    # 不在區間內

                    E_p_deg = min(abs(p - l), abs(p - r), abs(p - l + 360), abs(p - r + 360))
                    E_p = (E_p_deg * math.pi / 180) * radius
                    loss += E_p * (s[i, j] / 255.0)
    elif len(m) == 4:
        m0, m1, m2, m3 = m
        for i in range(h):
            for j in range(w):
                if mask[i, j] == 1:
                    p = h360[i, j]
                    l1 = (m0 + a) % 360
                    r1 = (m0 + m1 + a) % 360
                    l2 = (m2 + a) % 360
                    r2 = (m2 + m3 + a) % 360

                    in1 = False
                    in2 = False
                    if r1 < l1:
                        in1 = (p >= l1 or p <= r1)
                    else:
                        in1 = (l1 <= p <= r1)

                    if r2 < l2:
                        in2 = (p >= l2 or p <= r2)
                    else:
                        in2 = (l2 <= p <= r2)

                    if in1 or in2:
                        continue

                    E_p_deg = min(abs(p - l1), abs(p - r1), abs(p - l2), abs(p - r2), abs(p - l1 + 360), abs(p - r1 + 360), abs(p - l2 + 360), abs(p - r2 + 360))
                    E_p = (E_p_deg * math.pi / 180) * radius
                    loss += E_p * (s[i, j] / 255.0) 
    return loss

def compute_min_a(M_dict, h360, s, mask):
    best_a = {}
    min_loss = {}
    for name, m in M_dict.items():
        #m_tuple = tuple(name)
        min_loss[name] = float('inf')
        best_a[name] = -1
        for a in range(0, 360, 1):
            loss = compute_loss(np.array(m), a, h360, s, mask)
            if loss < min_loss[name]:
                min_loss[name] = loss
                best_a[name] = a
    return min_loss, best_a

if __name__ == "__main__":
    # 讀取圖片（以 BGR 格式）
    radius = 1
    mean = 0
    parser = argparse.ArgumentParser(description='Color Harmonization')
    parser.add_argument('--img_path', type=str, default=f'{IMG_PATH}/sample1.png', help='Path to the input image')
    parser.add_argument('--img_harmonized', type=str, default=f'{IMG_PATH}/sample1_foreground_padding.png', help='Object to be harmonized')
    parser.add_argument('--img_result_path', type=str, default=f'{IMG_RESULT_PATH}/test2', help='Path to save the result image')#2是增加了方向
    parser.add_argument('--mode', type=str, default=None, help='Template mode')
    parser.add_argument("--modify_direction", type=int, default=0, help="Modify the direction of the color shift")#0:不修改, 1: 修改
    parser.add_argument("--save", action="store_true", help="Save the result image")
    args = parser.parse_args()

    img_bgr = cv2.imread(args.img_path) #background
    img_foreground = cv2.imread(args.img_harmonized)#foreground
    #print("img_foreground: ", img_foreground)
    img_np = np.array(img_foreground)
    mask = np.any(img_foreground != [0, 0, 0], axis=-1).astype(np.uint8) #build a mask

    '''
    plt.figure(figsize=(8, 6))
    plt.imshow(mask)
    plt.show()
    '''

    # 將 BGR 圖片轉換成 HSV 格式
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    print("img_hsv.shape: ", img_hsv.shape)
    #print("img_hsv: ", img_hsv)

    h, s, v = cv2.split(img_hsv)

    # 把 H 從 0~179 映射到 0~360, 由於openCV 的 H 是 0~179，所以要乘以 2
    h360 = h.astype(float) * 2

    #[start angle, angle length, another start angle, another angle length]
    i, V, T = [351.0, 18.0], [313.2, 93.6], [0.0, 180.0] 
    L, I, Y, X = [351.0, 18.0, 50.4, 79.2], [351.0, 18.0, 171.0, 18.0], [313.2, 93.6, 171.0, 18.0], [313.2, 93.6, 133.2, 93.6]
    #M = [i, V, T, L, I, Y, X] #M = [i, V, T, L, I, Y, X]
    M_dict = {
    "i": i,
    "V": V,
    "T": T,
    "L": L,
    "I": I,
    "Y": Y,
    "X": X
    }

    ##進行對每個T_m選a的行為 
    best_a = {}
    min_loss = {}
    min_loss, best_a = compute_min_a(M_dict, h360, s, mask)
    min_val = min(min_loss.values())
    min_key = min(min_loss, key=min_loss.get)
    print("min_loss: ", min_loss)
    print("best_a: ", best_a)
    print("min_val: ", min_val)
    print("min_key: ", min_key)      

    new_h360 = h360.copy()
    degree_diff = np.zeros(h360.shape, dtype=np.float32)
    if args.mode in M_dict:
        template = (args.mode, M_dict[args.mode], best_a[args.mode]) #(template_name, template_list, template_a)
    else:
        template = (min_key, M_dict[min_key], best_a[min_key])
    print("template: ", template)
    center, arc_width, degree_diff, direction = compute_result(h360, s, mask, template, new_h360, degree_diff)
    print("direction: ", direction)
    '''
    for i in range(img_bgr.shape[0]):
        for j in range(img_bgr.shape[1]):
            if mask[i, j] == 0: #background
                if(len(template[1]) == 2):
                    m0 = (img_bgr[i, j, 0] + template[1][0] + template[2]) % 360
    '''
    for i in range(img_bgr.shape[0]):
        for j in range(img_bgr.shape[1]):
            if mask[i, j] == 0:
                p = new_h360[i, j]
                pdf_value = (degree_diff[i,j] * math.pi /180 ) * radius
                #print("val: ", degree_diff[i,j])
                #print("value: ", pdf_value)
                #print("center[i, j]: ", center[i, j])
                #print("arc_width[i, j]: ", arc_width[i, j])
                #print("pdf_value: ", pdf_value)
                #print("pdf_val: ", gaussian(pdf_value, (arc_width[i, j]/2)))
                #print("arc_width[i, j]: ", arc_width[i, j])
                if(args.modify_direction == 1):
                    if(direction[i, j] == 1):
                        new_h360[i, j] = (center[i,j] + ((arc_width[i, j] / 2)*(1 - gaussian(pdf_value, (arc_width[i, j]/2))))) % 360
                    else:
                        new_h360[i, j] = (center[i,j] - ((arc_width[i, j] / 2)*(1 - gaussian(pdf_value, (arc_width[i, j]/2))))) % 360
                else:
                    new_h360[i, j] = (center[i,j] + ((arc_width[i, j] / 2)*(1 - gaussian(pdf_value, (arc_width[i, j]/2))))) % 360
    #print("new_h360: ", new_h360)
    #shift the colors




    new_h180 = (new_h360 / 2).astype(np.uint8)

    img_result_hsv = cv2.merge([new_h180, s, v])
    img_result_bgr = cv2.cvtColor(img_result_hsv, cv2.COLOR_HSV2BGR)
    '''
    cv2.imshow("result", img_result_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    mode_str = f"_{args.mode}" if args.mode is not None else ""
    if args.mode == "I":
        mode_str = "__I"
    #save_path = args.img_result_path + mode_str + ".png"
    save_path = args.img_result_path + ".png"
    if args.save:
        cv2.imwrite(save_path, img_result_bgr)

    ##選擇最小的T_m
