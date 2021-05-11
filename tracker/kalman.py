import cv2
import numpy as np
import  math
from tracker.video_helper import VideoHelper

class KalmanFilter(object):
    def __init__(self,videoHelper):
        self.dynamParamsSize = 6
        self.measureParamSize =4
        self.kalman = cv2.KalmanFilter(dynamParams = self.dynamParamsSize,
                                       measureParams = self.measureParamSize)
        self.first_run = True
        dT = 1./videoHelper.frame_fps
        self.kalman.transitionMatrix = np.array([[1, 0, dT, 0, 0, 0],   #xc
                                                 [0, 1, 0, dT, 0, 0],   #xy
                                                 [0, 0, 1, 0, 0, 0],    #vx
                                                 [0, 0, 0, 1, 0, 0],    #vy
                                                 [0, 0, 0, 0, 1, 0],    #w
                                                 [0, 0, 0, 0, 0, 1]], np.float32)   #h
        # Measurement Matrix
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[0.01, 0, 0, 0, 0, 0],
                                                [0, 0.01, 0, 0, 0, 0],
                                                [0, 0, 5.0, 0, 0, 0],   #vx 的不确定性 大
                                                [0, 0, 0, 5.0, 0, 0],   #vy 的不确定性 大
                                                [0, 0, 0, 0, 0.01, 0],
                                                [0, 0, 0, 0, 0, 0.01]], np.float32)

        self.kalman.measurementNoiseCov = np.array([[0.1, 0, 0, 0],
                                                    [0, 0.1, 0, 0],
                                                    [0, 0, 0.1, 0],
                                                    [0, 0, 0, 0.1]], np.float32)
                                        # np.eye(4, dtype=np.float32) * 0.1


    def get_predicted_bbox(self):
        predicted_res = self.kalman.predict().T[0]
        predicted_bbox = self.get_bbox_from_kalman_form(predicted_res)
        return predicted_bbox
    def get_bbox_from_kalman_form(self,kalman_from):
        xc = kalman_from[0]
        yc = kalman_from[1]
        w = kalman_from[4]
        h = kalman_from[5]
        l = math.ceil(xc-w/2.)
        r = math.ceil(xc+w/2.)
        t = math.ceil(yc - h / 2.)
        b = math.ceil(yc + h / 2.)
        return [l, r, t, b]
    def correct(self,bbox):
        w = bbox[1] - bbox[0] + 1       # 2nd, 3rd —> length = 2nd + 3rd = 2 pixels = 3 - 2 + 1
        h = bbox[3] - bbox[2] + 1
        xc = int(bbox[0] + w / 2.)
        yc = int(bbox[2] + h / 2.)

        measurement = np.array([[xc, yc, w, h]], dtype=np.float32).T
        if self.first_run:
            self.kalman.statePre = np.array([measurement[0],measurement[1],
                                             [0],[0],
                                             measurement[2],measurement[3]],dtype=np.float32)
            self.first_run = False
        corrected_res = self.kalman.correct(measurement).T[0]
        corrected_bbox = self.get_bbox_from_kalman_form(corrected_res)
        return corrected_bbox

