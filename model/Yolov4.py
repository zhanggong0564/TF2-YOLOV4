from model.YOLONeck import PANet
from model.backbone.CSPDarknet53 import CSPDarkNet53
from model.backbone.mobilenetv2 import Mobilnetv2
from tensorflow.keras.layers import *
from  utils__.utils import *
from model.Tiny.yolov4Tiny import YoloV4Tiny


class Yolov4(tf.keras.Model):
    def __init__(self,number_class,anchors,masks):
        super(Yolov4, self).__init__()
        self.backbone = CSPDarkNet53()
        # self.backbone = Mobilnetv2()
        self.PANet = PANet(len(anchors[masks[0]])*(5+number_class))
        self.anchors = anchors
        self.number_class = number_class
        self.masks =masks
        self.boxes_0 = Lambda(Lambda(lambda x: yolo_boxes(x, anchors[masks[0]],number_class),
                             name='yolo_boxes_0'))
        self.boxes_1 = Lambda(Lambda(lambda x: yolo_boxes(x, anchors[masks[1]],number_class),
                             name='yolo_boxes_0'))
        self.boxes_2 = Lambda(Lambda(lambda x: yolo_boxes(x, anchors[masks[2]],number_class),
                             name='yolo_boxes_0'))
        self.outputs = Lambda(lambda x: yolo_nms(x,anchors, masks,number_class),
                             name='yolo_nms')
    # @tf.function
    def call(self, inputs,training=None):
        x3,x2,x1 = self.backbone(inputs,training = training)
        outputs = self.PANet([x3,x2,x1],training)#out1,out2,out3
        if training:
            return outputs
        else:
            boxes_0 = self.boxes_0(outputs[0])
            boxes_1 = self.boxes_1(outputs[1])
            boxes_2 = self.boxes_2(outputs[2])
            outputs = self.outputs((boxes_0[1:4], boxes_1[1:4], boxes_2[1:4]))#objectness, class_probs,bbox,
            return outputs

# def YOLOV4Tiny(inputs,anchors,masks,classes,training):
#     x = inputs = Input(inputs, name='input')
#     if training:
#         return YoloV4Tiny(inputs,num_anchors=len(masks[0]),num_class=classes)
#     else:
#         output_0,output_1 = YoloV4Tiny(inputs,num_anchors=len(masks[0]),num_class=classes)(x)
#         boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
#                          name='yolo_boxes_0')(output_0)
#         boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
#                          name='yolo_boxes_1')(output_1)
#         outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
#                          name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
#         return tf.keras.Model(inputs,outputs,name = 'yolov4_tiny')

if __name__ == '__main__':
    import numpy as np

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                             (59, 119), (116, 90), (156, 198), (373, 326)],
                            np.float32)
    yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    x = tf.random.uniform((1,416,416,3))
    model = Yolov4(4,yolo_anchors,yolo_anchor_masks,)
    # model = YOLOV4Tiny(inputs=(416,416,3),anchors=yolo_anchors,masks=yolo_anchor_masks,classes=4,training=True)
    model.build((1, 416, 416, 3))
    # model.load_weights(r'H:\yolov4\TF2-YOLOV4\yolov4_mobilenet.h5')
    #
    model.summary()
    y = model(x,training=True)
    for i in y:
        print(i.shape)
