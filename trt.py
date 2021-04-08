import cv2
import tensorflow as tf
import numpy as np
import config as cfg
import time
from model.Tiny.yolov4Tiny import YoloV4Tiny
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
if __name__ == '__main__':
    classes_dict = {0:'apple',1:'pear',2:'green',3:'orange'}
    # # classes_dict = {0:'apple',1:'pear',2:'green',3:'orange'}
    # #
    # yolov4_weight = 'yolov4_tiny.h5'
    # anchors = np.array([(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)])
    # anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])
    # model = YoloV4Tiny(inputs=(416, 416, 3), anchors=anchors, masks=anchor_masks, classes=cfg.num_class,
    #                    training=False)
    # model.load_weights(yolov4_weight)
    # model.save('tf2_savemodel',save_format='tf')
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    #
    # params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    # params._replace(precision_mode=trt.TrtPrecisionMode.FP16)
    # converter = trt.TrtGraphConverterV2(input_saved_model_dir='tf2_savemodel', conversion_params=params)
    # converter.convert()  # 完成转换,但是此时没有进行优化,优化在执行推理时完成
    # converter.save('trt_savedmodel')
    print('load model')
    t = time.time()
    saved_model_loaded = tf.saved_model.load(
        "trt_savedmodel", tags=[trt.tag_constants.SERVING])
    e = time.time()
    print('load trt model speed {}'.format(e-t))
    # 读取模型
    graph_func = saved_model_loaded.signatures[
        trt.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]  # 获取推理函数,也可以使用saved_model_loaded.signatures['serving_default']
    frozen_func = trt.convert_to_constants.convert_variables_to_constants_v2(
        graph_func)  # 将模型中的变量变成常量,这一步可以省略,直接调用graph_func也行
    capture = cv2.VideoCapture('data.mp4')
    while True:
        start = time.time()
        res, frame = capture.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (416, 416)) / 255.0
        tf_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        tf_tensor = tf.expand_dims(tf_tensor, 0)
        boxes, scores, classes, _ = frozen_func(tf_tensor)
        end = time.time()
        fps = 1 / (end - start)
        print(fps)
        for bbox, score, cls in zip(boxes[0], scores[0], classes[0]):
            if tf.reduce_sum(bbox) == 0:
                continue
            x1, y1, x2, y2 = bbox.numpy() * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv2.putText(frame, classes_dict[int(cls.numpy())] + ':' + str(score.numpy()), (int(x1), int(y1)),
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.9, (0, 225, 0))
            cv2.putText(frame, 'FPS:' + str(fps), (20, 20),
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.9, (0, 225, 0))
            cv2.imshow('t', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
    capture.release()
    cv2.destroyAllWindows()
