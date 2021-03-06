from __future__ import print_function
import time
from tracker.configs import Configs
from tracker.video_helper import VideoHelper
from tracker.multiple_object_controller import MultipleObjectController
from tracker.visualizer import Visualizer
from Yolodetect import YoloDetect
import config as cfg
import warnings
#忽略警告
warnings.filterwarnings('ignore')



def run():
    Detect = YoloDetect(cfg)
    # step 1: initialization
    # set configures: parameters are set here
    start_pre = time.time()
    configs = Configs()

    # detector: detect objects in a frame; return us detected results
    # detector = faceboxes()

    # video helper: read in / write out video rames
    video_helper = VideoHelper(configs)

    # object controller: objects are managed in this class
    object_controller = MultipleObjectController(configs, video_helper)
    print("Pre Time: ", (time.time() - start_pre) * 1000, " ms")

    # step 2: main loop
    cur_frame_counter = 0
    detection_loop_counter = 0
    while (video_helper.not_finished(cur_frame_counter)):
        print("####################################################### frame: ", cur_frame_counter)

        # get frame from video
        # frame is the raw frame, frame_show is used to show the result
        frame, frame_show = video_helper.get_frame()

        """ Detection Part """
        # if we detect every frame
        # (because detection now costs a lot so in order we can run in real time,
        # we may neglect some frames between detection)
        if configs.NUM_JUMP_FRAMES == 0:
            start_time_of_tracking = time.time()
            # detected results: [{'tag1':[bbx1]}, {'tag2':[bbx2]}, ..., {'tagn':[bbxn]}]
            detects = Detect.dectect_image(frame)
            # for detect in detects:
            # update current bbxes for each instance

            object_controller.update(detects, cur_frame_counter, frame)
            time_spend = time.time() - start_time_of_tracking
            print("Tracking Time: ", time_spend * 1000, " ms.")
        else:
            # we ignore to detect some frames
            if detection_loop_counter % configs.NUM_JUMP_FRAMES == 0:
                start_turn = time.time()
                # here we need to detect the frame
                detection_loop_counter = 0
                detects = Detect.dectect_image(frame)
                object_controller.update(detects, cur_frame_counter, frame)
                print("detection_loop_counter time span: ", (time.time() - start_turn) * 1000, " ms")
            else:
                # here we needn't to detect the frame
                start_turn_no_detect = time.time()
                object_controller.update_without_detection(cur_frame_counter, frame)
                #object_controller.update_still(cur_frame_counter)
                print("detection_loop_counter time span: ", (time.time() - start_turn_no_detect) * 1000, " ms")
        # 可视化
        visualizer = Visualizer(configs)
        show_temporal_information = True
        k = visualizer.drawing_tracking(frame_show, object_controller.instances, cur_frame_counter, show_temporal_information)
        if k==27:
            break
        cur_frame_counter += 1
        detection_loop_counter += 1
        print()


    video_helper.end()


if __name__ == "__main__":
    run()

