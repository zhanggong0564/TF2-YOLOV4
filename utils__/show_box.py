import numpy as np
import cv2
from utils__.utils import get_aim
from utils__.viduslizer import *


'''
1.得到了 image, boxes, labels, probs, class_labels
2.根据probs的高低阈值筛选
返回box和scores和每个框的类别
3.opencv将box和分数和类别画到image上并且返回物体的box和分数
'''
'''
果子类别对应区间
[1-100]苹果
[101-200]橙子
[201-300]梨子
[301-400]青苹果

'''

colors = [
    (0,255,255),
    (0,255,0),
    (255,0,0),
    (0,155,165)
]
font = cv2.FONT_HERSHEY_COMPLEX

class_ifo = {
    'apple':0,
    'pear':1,
    'green apple':2,
    'orange':3
}
no_grasp = "can't grasp"

def visualize_boxes(image, boxes, labels, probs,class_labels,color_intrin_part,aligned_depth_frame):
    category_index = {}
    for id_, label_name in enumerate(class_labels):
        category_index[id_] = {"name": label_name}
    box_info = find_box(boxes, labels, probs,category_index)
    show_image = draw_box(image,box_info,color_intrin_part,aligned_depth_frame)
    return show_image

def find_box(boxes, classes, scores,category_index,min_score_thresh=0.6):
    box_info = {}
    box_list = []
    class_list  = []
    scores_list = []
    sorted_ind = np.argsort(-scores)
    boxes = boxes[sorted_ind]  # 分数索引从大到小的框
    scores = scores[sorted_ind]  # 从大到小的分数
    classes = classes[sorted_ind]

    for i in range(min(20, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            s = scores[i]
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
            box_list.append(box)
            class_list.append(class_name)
            scores_list.append(s)
    box_info['box'] = box_list
    box_info['class'] = class_list
    box_info['scores'] = scores_list
    return box_info

def draw_box(image,box_info,color_intrin_part,aligned_depth_frame,line_thickness=None):
    show_image = image.copy()
    H,W,c = show_image.shape
    box_list= box_info['box']
    class_list = box_info['class']
    scores_list=box_info['scores']
    for index,box in enumerate(box_list):
        x0,y0,x1,y1 = box
        x0 = int(x0*W)
        y0 = int(y0*H)
        x1 = int(x1*W)
        y1 = int(y1*H)
        target_xyz_true,w,h = xy2xyz(x0,y0,x1,y1,color_intrin_part,aligned_depth_frame)
        _class = class_ifo[class_list[index]]
        score = scores_list[index]
        color = np.random.randint(0, 255, (1, 3))[0].tolist()


        border = h if w >= h else w
        draw_tag(show_image,class_list[index],x0,y0)
        draw_bbx(show_image,x0,x1,y0,y1,color)
        draw_corner(show_image, x0,x1,y0,y1, border, color)

        if np.sum(target_xyz_true)==0.0:
            text1 = "no depth info"
        else:
            text1 = str(target_xyz_true)
        draw_tag(show_image,text1,x0,y1+30)
        cv2.circle(show_image,((x0+x1)//2,(y0+y1)//2),10,color,-1,lineType=cv2.LINE_AA)
    return show_image

def xy2xyz(x0,y0,x1,y1,color_intrin_part,aligned_depth_frame):
    w = x1 - x0
    h = y1 - y0
    stride_w = w // 3
    stride_h = h // 3

    new_x0 = x0 + stride_w
    new_y0 = y0 + stride_h

    new_x1 = x1 - stride_w
    new_y1 = y1 - stride_h
    xyz_list = []
    loopx = range(new_x0, new_x1)
    loopy = range(new_y0, new_y1)
    for xc, yc in zip(loopx, loopy):
        target_xyz_true = get_aim(xc, yc, color_intrin_part, aligned_depth_frame)
        if target_xyz_true[2] != 0.0:
            xyz_list.append(target_xyz_true)
    # target_depth = aligned_depth_frame.get_distance(center_x, center_y)
    if xyz_list:
        mean_xyz = np.mean(xyz_list, 0)
    else:
        mean_xyz = np.array(xyz_list)

    def _round(x):
        return round(x, 3)

    if mean_xyz.any():
        target_xyz_true = list(map(_round, mean_xyz))
    else:return None
    return target_xyz_true,w,h
