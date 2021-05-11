import cv2
import tensorflow as tf
import pyrealsense2 as rs
import argparse
import socket
import numpy as np
import pickle
from model.Tiny.yolov4Tiny import YoloV4Tiny
import threading
import serial
import struct

from utils__.show_box import visualize_boxes

class WenSerber:
    def __init__(self,weight_path,addr,port,min_threshold,max_threshold):
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
        s = socket.socket()
        s.bind((addr,port))
        s.listen()
        self.s = s
        self.a = pickle.dumps(np.array([[0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0]]), protocol=0)
        self.b = pickle.dumps(np.array([[-1, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1]]), protocol=0)
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
    def get_cxcy(self,boxes,class_inds,cl=None):
        box = boxes[class_inds == cl][0]
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        return cx,cy
    def get_aim(self,cx,cy,color_intrin_part,aligned_depth_frame):
        # cxcy = np.array(list(show_dict.values()))
        # cx,cy = cxcy[...,0],cxcy[...,1]
        target_depth = aligned_depth_frame.get_distance(int(cx), int(cy))
        if target_depth==0.0:
            raise Exception("Invaild target_depth",target_depth)
        if  self.min_threshold<=target_depth<=self.max_threshold:#修改
            target_xyz_true = [(cx - color_intrin_part[0]) * target_depth / color_intrin_part[2],
                               (cy - color_intrin_part[1]) * target_depth / color_intrin_part[3],
                               target_depth]
            try:
                exter  = self.read_yaml('eyeToHand_calibration.yaml')
            except:
                print("eyeToHand_calibration.yaml don't exit")
            # aim = np.eye(3, 3)
            # aim = np.concatenate([aim, np.array(target_xyz_true).reshape(3, 1)], 1)
            # aim = np.concatenate([aim, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            # output = exter.dot(aim)
            # output[0,3] = output[0,3]-0.025#buchang
            # output[1,3] = output[1,3]-0.01
            # print(output)
            # output = pickle.dumps(output,protocol=0)
            return target_xyz_true
        else:
            output =self.a
            return output,None

    def read_yaml(self,path):
        infile = cv2.FileStorage(path,cv2.FILE_STORAGE_READ)
        rotate = infile.getNode("rotate").mat()
        translation = infile.getNode("translation").mat()
        temp = np.concatenate([rotate,translation],-1)
        eye = [
            [0,0,0,1]
        ]
        eye = np.array(eye)
        out = np.concatenate([temp,eye],0)
        return out
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
            '''
            boxes :array([[0.3554128 , 0.53700095, 0.47980157, 0.76449484],
                           [0.52972454, 0.580328  , 0.66029257, 0.8141017 ],
                           [0.4130515 , 0.2788775 , 0.5360351 , 0.50734454]], dtype=float32)
            scores:array([0.97277886, 0.9573462 , 0.92537   ], dtype=float32)
            classes:array([0, 0, 3], dtype=int32)
            '''
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
        self.s.close()
    def detection(self,conn,class_index):
        pass
    def process_cmd(self,msg,conn):
        if msg == "start":
            conn.send('start successed'.encode())
            print('detection start')
            while True:
                    print('--Please input the target to capture\n1:apple\n2:pear\n3:green apple\n4:orange\nstop:exit')
                    aim = conn.recv(1024).decode('utf-8')
                    print("this:{}".format(aim))
                    import time
                    start = time.time()
                    end = 0
                    #import pdb
                    #pdb.set_trace()
                    if aim == '1':
                        try:
                            self.detection(conn,0)
                            end = time.time()
                        except:
                            print("--This target was not detected--")
                            conn.send(self.a)
                            print('已发送')
                            end = time.time()
                            continue
                    elif aim == '2':
                        try:
                            self.detection(conn, 1)
                            end = time.time()
                            continue
                        except:
                            print("--This target was not detected--")
                            conn.send(self.a)
                            end = time.time()
                            continue
                    elif aim == '3':
                        try:
                            self.detection(conn, 2)
                            continue
                        except:
                            print("--This target was not detected--")
                            conn.send(self.a)
                            continue
                    elif aim == '4':
                        try:
                            self.detection(conn, 3)
                            continue
                        except:
                            print("--This target was not detected--")
                            conn.send(self.a)
                            continue

                    elif aim == 'stop':
                        try:
                            conn.send('stop successed'.encode())
                            cv2.destroyAllWindows()
                        except:
                            conn.send('stop failed'.encode())
                        break
                    elif not aim:
                        print('Disconnect')
                        break

                    else:
                        print('input erro,please again')
                        print(self.b)
                        conn.send(self.b)
                        continue
                    print('spend {} '.format(end-start))
        else:
            conn.send('start failed'.encode())
    def start(self):
        show = threading.Thread(target=self.detcetandshow)
        show.setDaemon(True)
        show.start()
        while True:
            conn, addr = self.s.accept()
            print("connected：", addr)
            print('please input："start" keep the working state')
            msg = conn.recv(1024).decode('utf-8')
            process = threading.Thread(target=self.process_cmd,args=(msg,conn))
            process.start()
            # if not
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--weight_path', type=str, default="./yolov4_tiny.h5")
    opt = parse.parse_args()
    addr = '127.0.0.1'
    port = 18080
    web = WenSerber(opt.weight_path,addr,port,0,8)
    web.start()

