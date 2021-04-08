import tensorflow as tf
from utils__.dataloader import YoloDataset
# from utils__.yololoss import yolo_loss
# from model.loss import yolo_loss
from model.Yolov4 import Yolov4, YOLOV4Tiny
import numpy as np
import config as cfg
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from utils__.loss import YoloLoss

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
# strategy = tf.distribute.MirroredStrategy(
#         devices=["/job:localhost/replica:0/task:0/device:GPU:0", "/job:localhost/replica:0/task:0/device:GPU:1"],
#         cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

strategy = tf.distribute.MirroredStrategy()

if __name__ == '__main__':
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))  # 输出设备数量

    with open(r'./2007_train.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = np.array(lines)
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(1)
    tf.random.set_seed(1)
    num_val = int(len(lines) * cfg.val_split)
    num_train = len(lines) - num_val
    if not cfg.yolotiny:
        anchors = np.array([(45, 62), (48, 55), (50, 52), (57, 65), (57, 64), (62, 58),
                            (63, 74), (67, 66), (73, 75)],
                           np.float32)
        anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    else:
        anchors = np.array([(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)],
                           np.float32)
        anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])

    dataset = YoloDataset(lines[:num_train], (416, 416), cfg.Bacthsize, anchors, cfg.num_class, cfg.use_mosico,
                          tiny=cfg.yolotiny)()
    dataset_val = YoloDataset(lines[num_train:], (416, 416), 2, anchors, cfg.num_class, tiny=cfg.yolotiny)()

    epoch_size = num_train / cfg.Bacthsize

    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    val_avg_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    loss_fn = [YoloLoss(anchors[i], classes=cfg.num_class) for i in anchor_masks]

    if not cfg.yolotiny:
        model = Yolov4(cfg.num_class, anchors, anchor_masks)
    else:
        model = YOLOV4Tiny(inputs=(416, 416, 3), anchors=anchors, masks=anchor_masks, classes=cfg.num_class,
                           training=True)
    # model.backbone.trainable = False

    lr_schedule = tf.keras.experimental.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=5 * epoch_size,
        t_mul=1.0,
        alpha=1e-2
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.restore(tf.train.latest_checkpoint(cfg.checkpoint_dir))
    model.compile(optimizer = optimizer,
                  loss=loss_fn,run_eagerly=True)
    model.fit(dataset,epochs=100,validation_data=dataset_val)



