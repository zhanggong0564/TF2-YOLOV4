import json
import os
from os import listdir, getcwd
from os.path import join
import os.path
import cv2

# rootdir = r'H:\data\cityscapes\leftImg8bit\train\zurich'  # 写自己存放图片的数据地址

def position(pos):
    # 该函数用来找出xmin,ymin,xmax,ymax即bbox包围框
    x = []
    y = []
    nums = len(pos)
    for i in range(nums):
        x.append(pos[i][0])
        y.append(pos[i][1])
    x_max = max(x)
    x_min = min(x)
    y_max = max(y)
    y_min = min(y)
    # print(x_max,y_max,x_min,y_min)
    b = (float(x_min), float(y_min), float(x_max), float(y_max))
    # print(b)
    return b


# pascal voc 标准格式
# < xmin > 174 < / xmin >
# < ymin > 101 < / ymin >
# < xmax > 349 < / xmax >
# < ymax > 351 < / ymax >

def convert(size, box):
    # 该函数将xmin,ymin,xmax,ymax转为x,y,w,h中心点坐标和宽高
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    print((x, y, w, h))
    return (x, y, w, h)


def convert_annotation(image_id,image,image_path,out_file):
    write_path = False
    classes = {'car': 0, 'traffic light': 1, 'pedestrian': 2}
    name = image_id.split('_')[0]
    # load_f = open("/home/ubuntu/PycharmProjects/city2pascal/source/train/tubingen/%s_gtFine_polygons.json" % (image_id), 'r')  # 导入json标签的地址
    load_f = open("H:/data/cityscapes/fine/train/%s/%s_gtFine_polygons.json" % (name,image_id),
                  'r')  # 导入json标签的地址
    load_dict = json.load(load_f)

    # keys=tuple(load_dict.keys())
    w = load_dict['imgWidth']  # 原图的宽，用于归一化
    h = load_dict['imgHeight']
    # print(h)
    objects = load_dict['objects']
    nums = len(objects)
    # print(nums)
    # object_key=tuple(objects.keys()
    cls_id = ''
    for i in range(0, nums):
        labels = objects[i]['label']
        # print(i)
        if (labels in ['person', 'rider']):
            if not write_path:
                out_file.write(image_path+" ")
                write_path = True
            # print(labels)
            pos = objects[i]['polygon']
            bb = position(pos)
            # bb = convert((w, h), b)
            # cv2.rectangle(image,(int(bb[0]),int(bb[1])),(int(bb[2]),int(bb[3])),(0,22,66),2)
            # cv2.imshow('cv',image)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            cls_id = 'pedestrian'  # 我这里把行人和骑自行车的人都设为类别pedestrian
            out_file.write(",".join([str(a) for a in bb])+" "+str(classes[cls_id])+" ")
            # print(type(pos))
        elif (labels in ['car', 'truck', 'bus', 'caravan', 'trailer']):
            if not write_path:
                out_file.write(image_path+" ")
                write_path = True
            # print(labels)
            pos = objects[i]['polygon']
            bb = position(pos)
            # bb = convert((w, h), b)
            # cv2.rectangle(image,(int(bb[0]),int(bb[1])),(int(bb[2]),int(bb[3])),(0,255,0),1)
            # cv2.imshow('cv',image)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            cls_id = 'car'  # 我这里把各种类型的车都设为类别car
            out_file.write(",".join([str(a) for a in bb])+" "+str(classes[cls_id])+" ")
        elif (labels in ['traffic light']):
            if not write_path:
                out_file.write(image_path+" ")
                write_path = True
            # print(labels)
            pos = objects[i]['polygon']
            bb = position(pos)
            # bb = convert((w, h), b)
            # cv2.rectangle(image,(int(bb[0]),int(bb[1])),(int(bb[2]),int(bb[3])),(0,255,0),1)
            # cv2.imshow('cv',image)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            cls_id = 'traffic light'  # 我这里把各种类型的车都设为类别car
            out_file.write(",".join([str(a) for a in bb])+" "+str(classes[cls_id])+" ")
    if write_path:
        out_file.write("\n")

    if cls_id == '':
        print('no label json:', "%s_gtFine_polygons.json" % (image_id))


def image_ids(rootdir):
    a = []
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            # print(filename)

            filename = filename[:-16]
            # filename = filename.strip('_leftImg8bit.png')
            a.append(filename)
    return a


if __name__ == '__main__':
    root = r'H:\data\cityscapes\leftImg8bit\train'
    rootdirs = [os.path.join(root, dirs) for dirs in os.listdir(root)]
    out_file = open('H:/data/cityscapes/dataset/zurich/train_leftImg8bit.txt', 'w')  # 输出标签的地址
    for rootdir in rootdirs:
        names = image_ids(rootdir)
        for image_id in names:
            print(image_id)
            image_path = os.path.join(rootdir,image_id+'_leftImg8bit.png')
            image =cv2.imread(image_path)
            convert_annotation(image_id,image,image_path,out_file)
