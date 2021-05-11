import tensorflow as tf
import cv2
import numpy as np
import random
import albumentations as A
from utils__.utils import get_mosico_random_data
import matplotlib.pyplot as plt
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] ="1"


class YoloDataset(object):
    def __init__(self,train_lines,image_size,batchsize,anchors,num_classes,mosaic =False,is_train=True,tiny=False):
        self.train_line = train_lines #['../025.jpg 374,207,426,250,1\n',...]
        self.image_size = image_size
        self.batchsize = batchsize
        self.mosaic =mosaic
        self.anchors = anchors
        self.num_classes = num_classes
        self.flag = True
        self.is_train = is_train
        self.tiny =tiny

    def __call__(self):
        def data_generator():
            for num,lines in enumerate(self.train_line) :
                choice = np.random.choice([1,2,3,4],1)
                if self.mosaic and choice[0]==1:
                    if num<(len(self.train_line)-5):#注意越界
                        image, box_data =  get_mosico_random_data(self.train_line[num:num+4],input_shape=self.image_size)
                        box_data = np.expand_dims(box_data,0)
                    else:
                        continue
                else:
                    image,box_data = self.get_random_data(lines)
                    # image = image/255.0
                    box_data = np.expand_dims(box_data,0)

                y_true = self.preprocess_true_boxes(box_data,self.image_size,self.anchors,self.num_classes,image)
                if self.tiny:
                    yield image/255.0,(y_true[0], y_true[1])
                else:
                    yield image/255.0,(y_true[0], y_true[1], y_true[2])
        if self.tiny:
            dataset = tf.data.Dataset.from_generator(generator=data_generator,
                                                     output_types=(tf.dtypes.float32, (
                                                    tf.dtypes.float32, tf.dtypes.float32)),
                                                     output_shapes=((416, 416, 3), ((13, 13, 3, 5 + self.num_classes),
                                                                                    (26, 26, 3, 5 + self.num_classes)))
                                                     )
        else:
            dataset = tf.data.Dataset.from_generator(generator=data_generator,
                                                     output_types=(tf.dtypes.float32,(tf.dtypes.float32,tf.dtypes.float32,tf.dtypes.float32)),
                                                     output_shapes=((416,416,3),((13, 13, 3, 5+self.num_classes),(26, 26, 3, 5+self.num_classes),(52, 52, 3, 5+self.num_classes)))
                                                     )
        dataset = dataset.shuffle(buffer_size=self.batchsize).prefetch(buffer_size=self.batchsize).batch(self.batchsize,drop_remainder=True)
        return dataset
    def get_random_data(self,annotation_line):
        line = annotation_line.split()
        image_src = cv2.imread(line[0])
        try:
            image_src.shape
        except:
            print("{}路径不存在".format(line[0]))
        image = cv2.cvtColor(image_src,cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image,(input_shape[0],input_shape[1]))
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        # import pdb
        # pdb.set_trace()

        split_bbox_info =np.hsplit(box,(4,6))
        bboxes = split_bbox_info[0]
        category_ids = split_bbox_info[1]
        image,box = self.augument(image, bboxes, category_ids)
        box_data = np.zeros((100,5))
        box_data[:len(box)] = box
        return image,box_data

    def augument(self,image,bbox,category_ids):
        category_ids = category_ids.reshape((1,-1))[0]
        transform = A.Compose([
            A.Resize(self.image_size[0],self.image_size[1],p=1),
            A.HorizontalFlip(p=0.5),
            A.RandomGamma(),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.RandomContrast(),
                A.RandomBrightness(),
                # A.ColorJitter(brightness=0.07, contrast=0.07,
                #               saturation=0.1, hue=0.1, always_apply=False, p=0.3),
                A.Cutout(),
                A.CLAHE(),
                A.Blur()
            ])
        ],bbox_params=A.BboxParams(format='pascal_voc',label_fields=['category_ids']))
        agument = transform(image=image,bboxes = bbox,category_ids = category_ids)
        image = agument['image']
        bboxs = agument['bboxes']
        category_ids = np.expand_dims(agument['category_ids'],axis=-1)
        # import pdb
        # pdb.set_trace()
        box = np.hstack((bboxs,category_ids))
        # 可视化
        # print(box)
        # for bb,id in zip(bboxs,category_ids):
        #     print(bb)
        #     print(id)
        #     bb =[int(i) for i in bb]
        #     cv2.rectangle(image,(bb[0],bb[1]),(bb[2],bb[3]),(0,255,0))
        #     cv2.putText(image, str(id), (bb[0],bb[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
        # cv2.imshow("image",cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        return image,box

    def preprocess_true_boxes(self,true_boxes, input_shape, anchors, num_classes,image):
        # import pdb
        # pdb.set_trace()

        assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
        # 一共有三个特征层数
        num_layers = len(anchors) // 3
        # -----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[142, 110], [192, 243], [459, 401]
        #   26x26的特征层对应的anchor是[36, 75], [76, 55], [72, 146]
        #   52x52的特征层对应的anchor是[12, 16], [19, 36], [40, 28]
        # -----------------------------------------------------------#


        # -----------------------------------------------------------#
        #   获得框的坐标和图片的大小
        # -----------------------------------------------------------#
        true_boxes = np.array(true_boxes, dtype='float32')#(n,100,5)
        # for bb in true_boxes[0]:
        #     print(bb)
        #     bb =[int(i) for i in bb]
        #     cv2.rectangle(image,(bb[0],bb[1]),(bb[2],bb[3]),(0,255,0))
        # cv2.imshow("image",cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        input_shape = np.array(input_shape, dtype='int32')#(416,416)
        # -----------------------------------------------------------#
        #   通过计算获得真实框的中心和宽高
        #   中心点(m,n,2) 宽高(m,n,2)
        # -----------------------------------------------------------#
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2#计算中心点
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]#计算宽和高


        # -----------------------------------------------------------#
        #   将真实框归一化到小数形式
        # -----------------------------------------------------------#
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        # m为图片数量，grid_shapes为网格的shape
        m = true_boxes.shape[0]
        #[array([13, 13], dtype=int32), array([26, 26], dtype=int32), array([52, 52], dtype=int32)]
        if num_layers==3:
            grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
            anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        else:
            grid_shapes = [input_shape // {0: 32, 1: 16}[l] for l in range(num_layers)]
            anchor_mask = [[3, 4, 5], [0, 1, 2]]
        # -----------------------------------------------------------#
        #   y_true的格式为[(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)]
        # -----------------------------------------------------------#
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                           dtype='float32') for l in range(num_layers)]


        # -----------------------------------------------------------#
        #   [9,2] -> [1,9,2]
        # -----------------------------------------------------------#

        anchors = np.expand_dims(anchors, 0)
        # anchor_maxes = anchors / 2.
        # anchor_mins = -anchor_maxes

        # -----------------------------------------------------------#
        #   长宽要大于0才有效
        # -----------------------------------------------------------#
        valid_mask = boxes_wh[..., 0] > 0
        for b in range(m):
            # 对每一张图进行处理
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0: continue
            # -----------------------------------------------------------#
            #   [n,2] -> [n,1,2]
            # -----------------------------------------------------------#
            wh = np.expand_dims(wh, -2)
            # box_maxes = wh / 2.
            # box_mins = -box_maxes

            # -----------------------------------------------------------#
            #   计算所有真实框和先验框的交并比
            #   intersect_area  [n,9]
            #   box_area        [n,1]
            #   anchor_area     [1,9]
            #   iou             [n,9]
            # -----------------------------------------------------------#
            intersect_area = np.minimum(wh[...,0],anchors[...,0])*np.minimum(wh[...,1],anchors[...,1])
            # intersect_mins = np.maximum(box_mins, anchor_mins)
            # intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            # intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            # intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]

            iou = intersect_area / (box_area + anchor_area - intersect_area)
            # -----------------------------------------------------------#
            #   维度是[n,] 感谢 消尽不死鸟 的提醒
            # -----------------------------------------------------------#
            best_anchor = np.argmax(iou, axis=-1)

            for t, n in enumerate(best_anchor):
                # -----------------------------------------------------------#
                #   找到每个真实框所属的特征层
                # -----------------------------------------------------------#
                for l in range(num_layers):
                    if n in anchor_mask[l]:
                        # -----------------------------------------------------------#
                        #   floor用于向下取整，找到真实框所属的特征层对应的x、y轴坐标
                        # -----------------------------------------------------------#
                        i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                        # -----------------------------------------------------------#
                        #   k指的的当前这个特征点的第k个先验框
                        # -----------------------------------------------------------#
                        k = anchor_mask[l].index(n)
                        # -----------------------------------------------------------#
                        #   c指的是当前这个真实框的种类
                        # -----------------------------------------------------------#
                        c = true_boxes[b, t, 4].astype('int32')#b=batch,t = 一张图片中的第几个框
                        # -----------------------------------------------------------#
                        #   y_true的shape为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
                        #   最后的85可以拆分成4+1+80，4代表的是框的中心与宽高、
                        #   1代表的是置信度、80代表的是种类
                        # -----------------------------------------------------------#
                        # import pdb
                        # pdb.set_trace()
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]#l表示三个不同尺度的特征图[0,1,2]分辨代表大中小 （x,y,w,h
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5 + c] = 1
        y_true = [np.squeeze(i) for i in y_true]
        return y_true
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

if __name__ == '__main__':
    val_split = 0.1
    with open('../data_info/train_val.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = np.array(lines)
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    anchors = get_anchors('../data_info/yolo_tiny.txt')

    dataset = YoloDataset(lines[:num_train],(416,416),4,anchors,4,mosaic=True,tiny=True)()
    for image,(y_true0,y_true1) in dataset:
        print(image.shape)
        # print(y_true0[...,4:5][y_true0[...,4:5]>0.5])
        print(y_true1.shape)
        print(y_true0.shape)
        # print(y_true2.shape)






