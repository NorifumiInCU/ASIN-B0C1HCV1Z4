import cv2
import matplotlib.pyplot as plt
import numpy as np
from basic import get_basic_data

dirname, basename, odir = get_basic_data(__file__)

# 画像を読み込んでリサイズ
img = cv2.imread("flower.jpg")
img = cv2.resize(img, (300, 169))
org_img = img.copy()
# 花びら間の境界を削る
img = cv2.resize(img, (300*12//100, 169*12//100))
img = cv2.resize(img, (300, 169))

# 色空間を二値化
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 花びらの色(ピンク、赤）領域のHSV範囲を指定
lower_red = np.array([140, 0, 0])
upper_red = np.array([180, 255, 255])

# HSV画像から赤い領域だけを抽出
red_mask = cv2.inRange(hsv, lower_red, upper_red)

# 元の画像とマスクを使って赤い領域だけを抽出
result = cv2.bitwise_and(img, img, mask=red_mask)
tmp = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
cv2.imwrite(f'{odir}/{basename}-gray.png', gray)

Lul='upper-left'
Lbr='bottom-right'
Lut='upper-threshold'
Llt='left-threshold'
Lbt='bottom-threshold'
Lrt='right-threshold'
Ldebug='debug'
test_list=[cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY]
extract_method_dic={
    cv2.THRESH_BINARY_INV:{
        # 左上最大、右下最小を抽出
        Lul:lambda v,t:max(v,t),
        Lut:0, Llt:0,
        Lbr:lambda v,t:min(v,t),
        Lbt:999, Lrt:999,
        Ldebug:f'THRESH_BINARY_INV'
    },
    cv2.THRESH_BINARY:{
        # 左上最小、右下最大を抽出
        Lul:lambda v,t:min(v,t),
        Lut:999, Llt:999,
        Lbr:lambda v,t:max(v,t),
        Lbt:0, Lrt:0,
        Ldebug:f'THRESH_BINARY'
    }
    }
for type in test_list:
    id='BIN_INV' if type==cv2.THRESH_BINARY_INV else 'BIN'
    bin_imgs=[]
    blur_imgs=[]
    blur_img_labels=[]
    get_ul_method=extract_method_dic[type][Lul]
    get_br_method=extract_method_dic[type][Lbr]
    debug_label=extract_method_dic[type][Ldebug]
    # size_list = [(3,3), (3,7), (7,3), (7,7), (7,9)]
    size_list = [(3,3), (7,9)]
    for sz in size_list:
        ## 画像をぼかす(平滑化)
        blur_img = cv2.GaussianBlur(gray, sz, 0)
        blur_imgs.append(blur_img)
        ## ２値化 threshold(image, threshold, judge white pixel val, convert algorithm)
        ### THRESH_BINARY_INV: if src(x,y)>threshold then dst(x,y)=0 else dst(x,y)=maxvalue(judge white pixel value)
        bin_img = cv2.threshold(blur_img, 140, 240, type)[1]
        bin_imgs.append(bin_img)
        blur_img_labels.append(f'{sz[0]},{sz[1]} blur')

    fig, axes = plt.subplots(len(size_list), 3)
    Lblur=0
    Lthreshold=1
    Lcnt=2
    for i in range(len(size_list)):
        img = org_img.copy()
        # 画面左側に二値化した画像を描画
        axes[i, Lblur].imshow(blur_imgs[i], cmap="gray")
        axes[i, Lblur].axis('off')
        axes[i, Lblur].set_title(blur_img_labels[i])
        axes[i, Lthreshold].imshow(bin_imgs[i], cmap="gray")
        axes[i, Lthreshold].axis('off')
        axes[i, Lthreshold].set_title('threshold')

        # 輪郭を抽出
        cnts = cv2.findContours(bin_imgs[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

        apply_left   = extract_method_dic[type][Llt]
        apply_upper  = extract_method_dic[type][Lut]
        apply_right  = extract_method_dic[type][Lrt]
        apply_bottom = extract_method_dic[type][Lbt]
        is_found = False
        for pt in cnts:
            x, y, w, h = cv2.boundingRect(pt)
            
            # 大きすぎたり小さすぎたりする領域を除去
            if w < 30 or w > 200:
                print(f'filtered:x:{x},y:{y},w:{w},h:{h} {debug_label}')
                continue
            else:
                print(f'x:{x},y:{y},w:{w},h:{h} {debug_label}')
            is_found = True
            apply_left=get_ul_method(x, apply_left)
            apply_upper=get_ul_method(y, apply_upper)
            apply_right=get_br_method(w, apply_right)
            apply_bottom=get_br_method(h, apply_bottom)
        if is_found:
            # 抽出した枠を描画
            print('rectangle:(',apply_left,apply_upper,apply_right,apply_bottom,')',' column:', i, debug_label)
            cv2.rectangle(img, (apply_left,apply_upper), (apply_left+apply_right, apply_upper+apply_bottom), (0, 255, 0), 2)
        # 画面右側に抽出結果を描画
        axes[i, Lcnt].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, Lcnt].axis('off')
        axes[i, Lcnt].set_title('Contours')
        print()

    name=f'{basename}-{id}'
    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(f"{odir}/{name}.png", dpi=100)
