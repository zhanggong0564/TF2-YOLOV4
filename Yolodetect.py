import numpy as np
from model.Yolov4 import Yolov4
import cv2
import tensorflow as tf
from model.Tiny.yolov4Tiny import YoloV4Tiny



class YoloDetect(object):
    def __init__(self,cfg):
        if cfg.yolotiny:
            yolov4_weight = '../yolov4_tiny.h5'
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
            self.model = Yolov4(cfg.num_class, anchors, masks=anchor_masks)
            self.model.build(input_shape=(None, 416, 416, 3))

        self.model.load_weights(yolov4_weight)

    def dectect_image(self,frame):
        classes_dict = {0: 'apple', 1: 'pear', 2: 'green', 3: 'orange'}
        res =[]
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
                obj_name = classes_dict[cls.numpy()]
                dic = {obj_name:[int(x1), int(x2),int(y1),int(y2)]}
                res.append(dic)
        return res
    @tf.function
    def get_pred(self,image_data):
        boxes, scores, classes = self.model(image_data)
        return boxes, scores, classes