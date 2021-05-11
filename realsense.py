import cv2
import tensorflow as tf
from model.Tiny.yolov4Tiny import YoloV4Tiny
import numpy as np
import config as cfg
from model.Yolov4 import Yolov4
import time
import pyrealsense2  as rs
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

@tf.function
def get_pred(image_data):
    boxes, scores, classes= model(image_data)
    # boxes, scores, classes, _ = model(image_data)
    return boxes, scores, classes
if __name__ == '__main__':
    classes_dict = {0:'apple',1:'pear',2:'green',3:'orange'}
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)


    if cfg.yolotiny:
        yolov4_weight = 'yolov4_tiny.h5'
        anchors = np.array([(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)])
        anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])
        model = YoloV4Tiny(inputs=(416, 416, 3), anchors=anchors, masks=anchor_masks, classes=cfg.num_class,
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

    model.load_weights(yolov4_weight)
    if cfg.pred_image:
        image_path = input('please input image path:')
        frame = cv2.imread(image_path)#'/media/zhanggong/3AB84F1E036B129D1/tf2/yolov4_tf2/yolov4-tf2-master/VOCdevkit/VOC2007/JPEGImages/60.jpg'
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(416,416))/255.0
        tf_tensor = tf.convert_to_tensor(image,dtype=tf.float32)
        tf_tensor = tf.expand_dims(tf_tensor,0)
        boxes, scores, classes = get_pred(tf_tensor)
        if scores.numpy().all():
            for bbox, score, cls in zip(boxes, scores, classes):
                if tf.reduce_sum(bbox) == 0.:
                    continue
                x1, y1, x2, y2 = bbox.numpy() * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                cv2.putText(frame, classes_dict[int(cls.numpy())] + ':' + str(score.numpy()), (int(x1), int(y1)),
                            cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.9, (0, 225, 0))
        cv2.imshow('t',frame)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        # capture = cv2.VideoCapture('H:/data.mp4')
        fps = 0.0
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            frame = color_image
            start = time.time()
            # res,frame = capture.read()
            h,w,c = frame.shape
            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image = cv2.resize(image,(416,416))/255.0
            tf_tensor = tf.convert_to_tensor(image,dtype=tf.float32)
            tf_tensor = tf.expand_dims(tf_tensor,0)
            boxes, scores, classes= get_pred(tf_tensor)
            end = time.time()
            fps = (fps+1/(end-start))/2
            print("fps:{:.2f}".format(fps))
            if scores.numpy().all():
                for bbox,score,cls in zip(boxes,scores,classes):
                    if tf.reduce_sum(bbox)==0.:
                        continue
                    x1,y1,x2,y2 = bbox.numpy()*[frame.shape[1],frame.shape[0],frame.shape[1],frame.shape[0]]
                    cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),1)
                    cv2.putText(frame,"{}:{:.2f}".format(classes_dict[int(cls.numpy())],score.numpy()),(int(x1),int(y1)),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,225,0))
                    cv2.putText(frame,'FPS:{:.2f}'.format(fps),(20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 225, 0))
            cv2.imshow('t',frame)
            k = cv2.waitKey(1)
            if k==27:
                break
        # capture.release()
        cv2.destroyAllWindows()
