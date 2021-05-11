import cv2
import tensorflow as tf
import pyrealsense2 as rs
import argparse
import numpy as np
from model.Tiny.yolov4Tiny import YoloV4Tiny
import threading
import serial
import struct
from utils__.show_box import xy2xyz
from utils__.show_box import visualize_boxes

class WenSerber:
    def __init__(self,weight_path,port,baudrate,min_threshold,max_threshold):
        super(WenSerber, self).__init__()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.pipeline.start(self.config)
        sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1]
        print(sensor.get_option(rs.option.exposure))
        self.align = rs.align(rs.stream.color)
        anchors = np.array([(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)])
        anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])
        self.model = YoloV4Tiny(inputs=(416, 416, 3), anchors=anchors, masks=anchor_masks, classes=4,
                           training=False)
        self.model.load_weights(weight_path)
        try:
            self.ser = serial.Serial(port=port, baudrate=baudrate)
        except:
            print("no serial")
        self.a = [0,0,0,0,0,0]
        self.min_threshold = min_threshold#修改
        self.max_threshold = max_threshold#修改
        self.classes = ['apple', 'pear', 'green apple', 'orange']
    @tf.function
    def predict(self,tensor_image):
        boxes, scores, classes = self.model(tensor_image)
        return boxes, scores, classes
    def get_img_tensor(self,image):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (416, 416)) / 255.0
        tf_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        tf_tensor = tf.expand_dims(tf_tensor, 0)
        return tf_tensor
    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        return color_frame,aligned_depth_frame
    def detcetandshow(self):
        while(True):
            color_frame, aligned_depth_frame = self.get_frame()
            color_image = np.asanyarray(color_frame.get_data())
            color_profile = color_frame.get_profile()
            cvsprofile = rs.video_stream_profile(color_profile)
            color_intrin = cvsprofile.get_intrinsics()
            color_intrin_part = [color_intrin.ppx, color_intrin.ppy, color_intrin.fx, color_intrin.fy]

            if not color_frame:
                assert color_frame is not None
                continue
            tensor_image = self.get_img_tensor(color_image)
            boxes, scores, classes = self.predict(tensor_image)
            boxes, scores, classes =  boxes.numpy(), scores.numpy(), classes.numpy()
            if scores.all() :
                show_image = visualize_boxes(image=color_image,
                                boxes=boxes,
                                labels=classes,
                                probs=scores,
                                class_labels=self.classes,
                                color_intrin_part=color_intrin_part,
                                aligned_depth_frame=aligned_depth_frame)
                cv2.namedWindow('video_test',0)
                cv2.resizeWindow('video_test',1920, 1080)
                cv2.imshow('video_test', show_image)
                k = cv2.waitKey(1)
                if k == 27:
                    cv2.destroyAllWindows()
                    break

            else:
                cv2.namedWindow('video_test',0)
                cv2.resizeWindow('video_test',1920, 1080)
                cv2.imshow('video_test', color_image)
                k = cv2.waitKey(10)
                if k == 27:
                    cv2.destroyAllWindows()
                    break
    def detection(self):
        while (True):
            color_frame, aligned_depth_frame = self.get_frame()
            color_image = np.asanyarray(color_frame.get_data())
            h,w,c = color_image.shape
            color_profile = color_frame.get_profile()
            cvsprofile = rs.video_stream_profile(color_profile)
            color_intrin = cvsprofile.get_intrinsics()
            color_intrin_part = [color_intrin.ppx, color_intrin.ppy, color_intrin.fx, color_intrin.fy]

            if not color_frame:
                assert color_frame is not None
                continue
            tensor_image = self.get_img_tensor(color_image)
            boxes, scores, classes = self.predict(tensor_image)
            boxes, scores, classes = boxes.numpy(), scores.numpy(), classes.numpy()
            scale = np.array([w,h,w,h])
            boxes_true = boxes*scale
            num_cls0 = 0
            num_cls1 = 100
            num_cls2 = 200
            num_cls3 = 300
            # import pdb
            # pdb.set_trace()
            for i,cls in enumerate(classes):
                box = boxes_true[i]
                #[0.068, 0.093, 0.474]
                target_xyz_true, _, _ = xy2xyz(int(box[0]), int(box[1]), int(box[2]), int(box[3]), color_intrin_part, aligned_depth_frame)
                if cls==0:
                    num_cls0 +=1
                    pack_data = [num_cls0] + target_xyz_true + [sum(target_xyz_true)]
                    # print(pack_data)
                    pack = struct.pack("<5f", *pack_data)
                    d = [i for i in pack]
                    data = bytearray(d)
                    self.ser.write(data)
                elif cls==1:
                    num_cls1+=1
                    pack_data = [num_cls1]+target_xyz_true+[sum(target_xyz_true)]
                    pack = struct.pack("<5f", *pack_data)
                    d = [i for i in pack]
                    data = bytearray(d)
                    self.ser.write(data)
                elif cls==2:
                    num_cls2+=1
                    pack_data = [num_cls2]+target_xyz_true+[sum(target_xyz_true)]
                    pack = struct.pack("<5f", *pack_data)
                    d = [i for i in pack]
                    data = bytearray(d)
                    self.ser.write(data)
                elif cls==3:
                    num_cls3+=1
                    pack_data = [num_cls3]+target_xyz_true+[sum(target_xyz_true)]
                    pack = struct.pack("<5f", *pack_data)
                    d = [i for i in pack]
                    data = bytearray(d)
                    self.ser.write(data)

    def start(self):
        show = threading.Thread(target=self.detcetandshow)
        show.setDaemon(True)
        show.start()

        process = threading.Thread(target=self.detection)
        process.start()
            # if not
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--weight_path', type=str, default="./yolov4_tiny.h5")
    opt = parse.parse_args()
    baudrate = 115200
    port = "/dev/ttyUSB0"
    web = WenSerber(opt.weight_path,port,baudrate,0,8)
    web.start()

