import cv2
import onnx
import onnxruntime
import keras2onnx
import tensorflow as tf
import numpy as np
import config as cfg
from model.Tiny.yolov4Tiny import YoloV4Tiny
import time

# yolov4_weight = 'yolov4_tiny.h5'
# anchors = np.array([(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)])
# anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])
# model = YoloV4Tiny(inputs=(416, 416, 3), anchors=anchors, masks=anchor_masks, classes=cfg.num_class,
#                    training=False)
# model.load_weights(yolov4_weight)
# onnx_model = keras2onnx.convert_keras(model,model.name)
# inputs = onnx_model.graph.input
# for input in inputs:
#     dim1 = input.type.tensor_type.shape.dim[0]
#     dim1.dim_value = 1
# model_name = 'yolo_tiny_model.onnx'
# onnx.save_model(onnx_model,model_name)
classes_dict = {0:'apple',1:'pear',2:'green',3:'orange'}
model = onnx.load_model('yolo_tiny_model.onnx')
onnx.checker.check_model(model)
session = onnxruntime.InferenceSession("yolo_tiny_model.onnx")
input_name = session.get_inputs()[0].name
label_name1 = session.get_outputs()[0].name
label_name2 = session.get_outputs()[1].name
label_name3 = session.get_outputs()[2].name
capture = cv2.VideoCapture('data.mp4')
num_frame = 0
while True:
    start = time.time()
    res, frame = capture.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (416, 416)) / 255.0
    # tf_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    tf_tensor = np.expand_dims(image, 0)
    s = time.time()
    bbox, scores, class_probs = session.run([label_name1,label_name2,label_name3],{input_name:tf_tensor.astype(np.float32)})

    # print()
    idxs = cv2.dnn.NMSBoxes(bbox,np.expand_dims(scores.sum(-1),-1),0.5,0.5)

    e  = time.time()
    fps = 1/(e-s)
    print(fps)
    for idx in idxs:
        x1, y1, x2, y2 =bbox[0][idx[0]]*[frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
        clas = class_probs[0][idx[0]]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        cv2.putText(frame, classes_dict[clas.argmax()]+str(fps), (int(x1), int(y1)),
                    cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.9, (0, 225, 0))
    cv2.imshow('e',frame)
    k = cv2.waitKey(10)
    if k ==27:
        break
cv2.destroyAllWindows()
capture.release()

