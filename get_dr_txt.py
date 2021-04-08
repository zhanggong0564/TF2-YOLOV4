# ----------------------------------------------------#
#   获取测试集的detection-result和images-optional
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#
import os
import tensorflow as tf
from tqdm import tqdm
from model.Tiny.yolov4Tiny import YoloV4Tiny
import numpy as np
import config as cfg
from model.Yolov4 import Yolov4
import cv2
import argparse




class YoloDetect(object):
    def __init__(self):
        if cfg.yolotiny:
            yolov4_weight = 'yolov4_tiny.h5'
            anchors = np.array([(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)])
            anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])
            self.model = YoloV4Tiny(inputs=(416, 416, 3), anchors=anchors, masks=anchor_masks, classes=cfg.num_class,
                               training=False)
            # image_shape = tf.keras.layers.Input(shape=(416, 416, 3))
            # model = yolo_body(inputs=image_shape,num_anchors=3,num_classes=3,anchors=anchors,masks=anchor_masks,training=False)
        else:
            yolov4_weight = 'yolov4.h5'
            anchors = np.array([(45, 62), (48, 55), (50, 52), (57, 65), (57, 64), (62, 58),
                                (63, 74), (67, 66), (73, 75)],
                               np.float32)
            anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
            model = Yolov4(cfg.num_class, anchors, masks=anchor_masks)
            model.build(input_shape=(None, 416, 416, 3))

        self.model.load_weights(yolov4_weight)

    def dectect_image(self,image_path,image_id):
        f = open("./input/detection-results/" + image_id + ".txt", "w")
        frame = cv2.imread(image_path)#'/media/zhanggong/3AB84F1E036B129D1/tf2/yolov4_tf2/yolov4-tf2-master/VOCdevkit/VOC2007/JPEGImages/60.jpg'
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(416,416))/255.0
        tf_tensor = tf.convert_to_tensor(image,dtype=tf.float32)
        tf_tensor = tf.expand_dims(tf_tensor,0)
        boxes, scores, classes = self.get_pred(tf_tensor)
        if scores.numpy().all():
            for bbox, score, cls in zip(boxes, scores, classes):
                if tf.reduce_sum(bbox) == 0.:
                    continue
                x1, y1, x2, y2 = bbox.numpy() * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
                f.write("%s %s %s %s %s %s\n"%(classes_dict[cls.numpy()],str(score.numpy()),str(int(x1)),str(int(y1)),str(int(x2)),str(int(y2))))
        f.close()
        return

    @tf.function
    def get_pred(self,image_data):
        boxes, scores, classes = self.model(image_data)
        # boxes, scores, classes, _ = model(image_data)
        return boxes, scores, classes

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h',help='use voc or txt')
    parser.add_argument('--voc',default=False,type=str2bool)
    args = parser.parse_args()
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    yolo = YoloDetect()
    classes_dict = {0: 'apple', 1: 'pear', 2: 'green', 3: 'orange'}


    if not os.path.exists("./input"):
        os.makedirs("./input")
    if not os.path.exists("./input/detection-results"):
        os.makedirs("./input/detection-results")
    if not os.path.exists("./input/images-optional"):
        os.makedirs("./input/images-optional")
    if args.voc:
        image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()
        for image_id in tqdm(image_ids):
            image_path = "./VOCdevkit/VOC2007/JPEGImages/" + image_id + ".jpg"
            # 开启后在之后计算mAP可以可视化
            # image.save("./input/images-optional/"+image_id+".jpg")
            yolo.dectect_image(image_path,image_id)
    if not args.voc:
        with  open('data_info/2007_test.txt') as f:
            txtinfo = f.readlines()
            for image_info in txtinfo:
                image_path = image_info.strip().split()[0]
                image_id = image_path.split('/')[-1].split('.')[0]
                yolo.dectect_image(image_path,image_id)
    print("Conversion completed!")
