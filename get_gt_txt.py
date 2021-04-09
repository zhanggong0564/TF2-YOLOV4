import sys
import os
import glob
import xml.etree.ElementTree as ET
import argparse

'''
！！！！！！！！！！！！！注意事项！！！！！！！！！！！！！
# 这一部分是当xml有无关的类的时候，下方有代码可以进行筛选！
'''
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h',help='use voc or txt')
    parser.add_argument('--voc',default=False,type=str2bool)
    args = parser.parse_args()
    classes_dict = {0: 'apple', 1: 'pear', 2: 'green', 3: 'orange'}
    with  open('data_info/2007_test.txt') as f:
        txtinfo = f.readlines()
    if args.voc:
        image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()
        for image_id in image_ids:
            with open("./input/ground-truth/"+image_id+".txt", "w") as new_f:
                root = ET.parse("VOCdevkit/VOC2007/Annotations/"+image_id+".xml").getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    '''
                    ！！！！！！！！！！！！注意事项！！！！！！！！！！！！
                    # 这一部分是当xml有无关的类的时候，可以取消下面代码的注释
                    # 利用对应的classes.txt来进行筛选！！！！！！！！！！！！
                    '''
                    # classes_path = 'model_data/voc_classes.txt'
                    # class_names = get_classes(classes_path)
                    # if obj_name not in class_names:
                    #     continue

                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

        print("Conversion completed!")

    if not args.voc:
        with  open('data_info/2007_test.txt') as f:
            txtinfo = f.readlines()
            for image_info in txtinfo:
                image_path,boxesinfos =image_info.strip().split()[0],image_info.strip().split()[1:]
                image_id = image_path.split('/')[-1].split('.')[0]
                with open("./input/ground-truth/" + image_id + ".txt", "w") as new_f:
                    for boxs in boxesinfos:
                        boxesinfo = boxs.split(',')
                        class_id = int(boxesinfo[-1])
                        left, top, right, bottom = int(boxesinfo[0]),int(boxesinfo[1]),int(boxesinfo[2]),int(boxesinfo[3])
                        obj_name = classes_dict[class_id]
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
            print("Conversion completed!")

