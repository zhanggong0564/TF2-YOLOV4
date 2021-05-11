import cv2
import tensorflow as tf
import math
import numpy as np
import math
from  matplotlib.colors import rgb_to_hsv,hsv_to_rgb
from functools import reduce
from tensorflow.keras import backend as K


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x, y, w, h))pre
    # box_2: (N, (x, y, w, h)) #中心点，wh true

    box_xy = box_2[...,:2]
    box_wh = box_2[...,2:]

    box_2_x1y1 = box_xy-box_wh/2
    box_2_x2y2 = box_xy+box_wh/2
    box_2 = tf.concat((box_2_x1y1,box_2_x2y2),-1)

    box1_xy = box_1[...,:2]
    box1_wh = box_1[...,2:]

    box_1_x1y1 = box1_xy-box1_wh/2
    box_1_x2y2 = box1_xy+box1_wh/2
    box_1 = tf.concat((box_1_x1y1,box_1_x2y2),-1)

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    box_1_area = box_1[..., 2] * box_1[..., 3]
    box_2_area = box_2[..., 2] * box_2[..., 3]


    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    # box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
    #     (box_1[..., 3] - box_1[..., 1])
    # box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
    #     (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)
def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    epsilon = 1e-8
    iou = intersect_area / tf.maximum(union_area, epsilon)

    # 计算中心的差距
    center_distance = tf.reduce_sum(tf.square(b1_xy - b2_xy), axis=-1)
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = tf.minimum(b1_mins, b2_mins)
    enclose_maxes = tf.maximum(b1_maxes, b2_maxes)
    enclose_wh = tf.maximum(enclose_maxes - enclose_mins, 0.0)
    # 计算对角线距离
    enclose_diagonal = tf.reduce_sum(tf.square(enclose_wh), axis=-1)

    ciou = iou - 1.0 * (center_distance) / tf.maximum(enclose_diagonal, epsilon)

    v = 4 * tf.square(tf.math.atan2(b1_wh[..., 0], tf.maximum(b1_wh[..., 1], epsilon)) - tf.math.atan2(b2_wh[..., 0],
                                                                                                         tf.maximum(
                                                                                                             b2_wh[
                                                                                                                 ..., 1],
                                                                                                             epsilon))) / (
                    math.pi * math.pi)
    alpha = v / tf.maximum((1.0 - iou + v), epsilon)
    ciou = ciou - alpha * v

    ciou = tf.expand_dims(ciou, -1)
    ciou = tf.where(tf.math.is_nan(ciou), tf.zeros_like(ciou), ciou)
    return ciou
def _smooth_labels(y_true, label_smoothing):
    num_classes = tf.cast(tf.shape(y_true)[-1],tf.float32)
    label_smoothing = tf.constant(label_smoothing, tf.float32)
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue

            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox


def get_mosico_random_data(annotation_line, input_shape, random=True, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    h, w = input_shape
    min_offset_x = 0.4
    min_offset_y = 0.4
    scale_low = 1 - min(min_offset_x, min_offset_y)
    scale_high = scale_low + 0.2

    image_datas = []
    box_datas = []
    index = 0

    place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
    place_y = [0, int(h * min_offset_y), int(w * min_offset_y), 0]
    # print('......')
    # print(annotation_line)
    for line in annotation_line:
        # 每一行进行分割
        line_content = line.split()
        # print(line_content)
        # 打开图片
        image = cv2.imread(line_content[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 图片的大小
        ih, iw, _ = image.shape
        # 保存框的位置
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

        # image.save(str(index)+".jpg")
        # 是否翻转图片
        flip = rand() < .5
        if flip and len(box) > 0:
            image = cv2.flip(image, 1)
            box[:, [0, 2]] = iw - box[:, [2, 0]]

        # 对输入进来的图片进行缩放
        new_ar = w / h
        scale = rand(scale_low, scale_high)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = cv2.resize(image, (nw, nh))

        # 进行色域变换
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image = hsv_to_rgb(x)

        image = (image * 255).astype(np.uint8)
        # 将图片进行放置，分别对应四张分割图片的位置
        dx = place_x[index]
        dy = place_y[index]
        new_image = np.zeros((w, h, 3), dtype=np.uint8)
        new_image[dy:np.clip((dy + nh), 0, h), dx:np.clip((dx + nw), 0, w), :] = image[:(np.clip((dy + nh), 0, h) - dy),
                                                                                 :(np.clip((dx + nw), 0, w) - dx), :]
        image_data = new_image / 255

        # Image.fromarray((image_data*255).astype(np.uint8)).save(str(index)+"distort.jpg")

        index = index + 1
        box_data = []
        # 对box进行重新处理
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box

        image_datas.append(image_data)
        box_datas.append(box_data)

        img = (image_data * 255).astype(np.uint8)
        for j in range(len(box_data)):
            thickness = 3
            left, top, right, bottom = box_data[j][0:4]
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
        # cv2.imshow(str(left), img)
        # cv2.waitKey()

    # 将图片分割，放在一起
    cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
    cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

    new_image = np.zeros([h, w, 3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    # 对框进行进一步的处理
    new_boxes = merge_bboxes(box_datas, cutx, cuty)
    new_boxes = np.array(new_boxes)
    box_data_info = np.zeros((100, 5))
    box_data_info[:len(new_boxes)] = new_boxes[:100]

    return new_image, box_data_info
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    # input_shape = np.array(input_shape)
    num_anchors = len(anchors)
    #---------------------------------------------------#
    #   [1, 1, 1, num_anchors, 2]
    #---------------------------------------------------#
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    #---------------------------------------------------#
    #   获得x，y的网格
    #   (13, 13, 1, 2)
    #---------------------------------------------------#
    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    #---------------------------------------------------#
    #   将预测结果调整成(batch_size,13,13,3,85)
    #   85可拆分成4 + 1 + 80
    #   4代表的是中心宽高的调整参数
    #   1代表的是框的置信度
    #   80代表的是种类的置信度
    #---------------------------------------------------#
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    #---------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy对应框的中心点
    #   box_wh对应框的宽和高
    #---------------------------------------------------#
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[...,::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[...,::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    #---------------------------------------------------------------------#
    #   在计算loss的时候返回grid, feats, box_xy, box_wh
    #   在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    #---------------------------------------------------------------------#
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs
def box_iou(b1, b2):
    # 13,13,3,1,4
    # 计算左上角的坐标和右下角的坐标
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # 1,n,4
    # 计算左上角和右下角的坐标
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 计算重合面积
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5,
              eager=False):
    if eager:
        image_shape = K.reshape(yolo_outputs[-1], [-1])
        num_layers = len(yolo_outputs) - 1
    else:
        # ---------------------------------------------------#
        #   获得特征层的数量，有效特征层的数量为3
        # ---------------------------------------------------#
        num_layers = len(yolo_outputs)

    # -----------------------------------------------------------#
    #   13x13的特征层对应的anchor是[81,82], [135,169], [344,319]
    #   26x26的特征层对应的anchor是[23,27], [37,58], [81,82]
    # -----------------------------------------------------------#
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    # -----------------------------------------------------------#
    #   这里获得的是输入图片的大小，一般是416x416
    # -----------------------------------------------------------#
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    # -----------------------------------------------------------#
    #   对每个特征层进行处理
    # -----------------------------------------------------------#
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # -----------------------------------------------------------#
    #   将每个特征层的结果进行堆叠
    # -----------------------------------------------------------#
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # -----------------------------------------------------------#
        #   取出所有box_scores >= score_threshold的框，和成绩
        # -----------------------------------------------------------#
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        # -----------------------------------------------------------#
        #   非极大抑制
        #   保留一定区域内得分最大的框
        # -----------------------------------------------------------#
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        # -----------------------------------------------------------#
        #   获取非极大抑制后的结果
        #   下列三个分别是
        #   框的位置，得分与种类
        # -----------------------------------------------------------#
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape):
    #-----------------------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy : -1,13,13,3,2;
    #   box_wh : -1,13,13,3,2;
    #   box_confidence : -1,13,13,3,1;
    #   box_class_probs : -1,13,13,3,80;
    #-----------------------------------------------------------------#
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    #-----------------------------------------------------------------#
    #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
    #   因此生成的box_xy, box_wh是相对于有灰条的图像的
    #   我们需要对齐进行修改，去除灰条的部分。
    #   将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
    #-----------------------------------------------------------------#

    box_yx = box_xy
    box_hw = box_wh
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)

    # input_shape = K.cast(input_shape, K.dtype(box_yx))
    # image_shape = K.cast(image_shape, K.dtype(box_yx))

    boxes =  K.concatenate([
        box_mins[..., 0:1] ,  # y_min
        box_mins[..., 1:2] ,  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])
    #-----------------------------------------------------------------#
    #   获得最终得分和框的位置
    #-----------------------------------------------------------------#
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

def get_aim(cx,cy,color_intrin_part,aligned_depth_frame):
    target_depth = aligned_depth_frame.get_distance(int(cx), int(cy))
    # if target_depth==0.0:
    #     raise Exception("Invaild target_depth",target_depth)
    target_xyz_true = [(cx - color_intrin_part[0]) * target_depth / color_intrin_part[2],
                       (cy - color_intrin_part[1]) * target_depth / color_intrin_part[3],
                       target_depth]
    return target_xyz_true
         










