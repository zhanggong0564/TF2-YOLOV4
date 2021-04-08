import tensorflow as tf
from utils__.dataloader import YoloDataset
from utils__.lossv2 import yolo_loss
# from model.loss import yolo_loss
from model.Yolov4 import Yolov4
from model.Tiny.yolov4Tiny import YoloV4Tiny
import numpy as np
import config as cfg
import os
import time
from model.Tiny.keras import yolo_body
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
strategy = tf.distribute.MirroredStrategy(
        devices=["/job:localhost/replica:0/task:0/device:GPU:0", "/job:localhost/replica:0/task:0/device:GPU:1"],
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

# strategy = tf.distribute.MirroredStrategy()
#单GPU
# @tf.function
# def train_step(image,label):
#     with tf.GradientTape() as t:
#         pred = model(image,training=True)
#         regularization_loss = tf.reduce_sum(model.losses)
#         args = [*pred] + [*label]
#         loss_value = yolo_loss(args,anchors,num_classes=4,normalize=False)
#         total_loss = tf.reduce_sum(regularization_loss)+loss_value
#         gradients = t.gradient(total_loss,model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients,model.trainable_variables))
#         return loss_value
with strategy.scope():
    def train_step(image,labels):
        with tf.GradientTape() as t:
            preds = model(image,training=True)
            total_loss =compute_loss(preds,labels)
        gradients = t.gradient(total_loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        return total_loss


    def val_step(image, labels):
        preds = model(image, training=True)
        total_loss = compute_loss(preds, labels)
        return total_loss
    def compute_loss(preds, labels):
        # tal_loss = 0
        # for pred, label, loss in zip(preds, labels, loss_fn):
        #     loss_val = loss(label, pred)
        #     tal_loss += loss_val
        args = preds + [labels[0],labels[1]]
        tal_loss = yolo_loss(args, anchors, cfg.num_class, label_smoothing=cfg.label_smoothing, normalize=False)
        return tf.nn.compute_average_loss(tal_loss, global_batch_size=cfg.Bacthsize)
    @tf.function
    def distributed_train_step(image, label):
        per_replica_losses = strategy.experimental_run_v2(train_step,
                                                          args=(image, label))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)
    @tf.function
    def distributed_val_step(image, label):
        per_replica_losses = strategy.experimental_run_v2(val_step,
                                                          args=(image, label))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)




if __name__ == '__main__':
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))  # 输出设备数量
    with open(cfg.data_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = np.array(lines)
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(1)
    tf.random.set_seed(1)
    num_val = int(len(lines) * cfg.val_split)
    num_train = len(lines) - num_val
    if not cfg.yolotiny:
        anchors = np.array([(45, 62), (48, 55), (50, 52), (57, 65),(57, 64),(62, 58),
                                 (63, 74), (67, 66), (73, 75)],
                                np.float32)
        anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    else:
        anchors = np.array([(10,14),(23,27),(37,58),(81,82),(135,169),(344,319)],
                           np.float32)
        anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])

    dataset = YoloDataset(lines[:num_train], (416, 416), cfg.Bacthsize, anchors, cfg.num_class,cfg.use_mosico,tiny=cfg.yolotiny)()
    dataset_val = YoloDataset(lines[num_train:], (416, 416), 8, anchors, cfg.num_class,tiny=cfg.yolotiny)()
    dataset = strategy.experimental_distribute_dataset(dataset)
    dataset_val =strategy.experimental_distribute_dataset(dataset_val)
    epoch_size = num_train/cfg.Bacthsize
    with strategy.scope():

        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        val_avg_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        loss_fn = [YoloLoss(anchors[i],classes=cfg.num_class,label_smoothing=cfg.label_smoothing) for i in anchor_masks]

        if not cfg.yolotiny:
            model = Yolov4(cfg.num_class, anchors, anchor_masks)
            model.build((8,416,416,3))
            print('load model yolo4_weights')
            model.load_weights('yolov4.h5',by_name=True,skip_mismatch=True)
        else:
            model = YoloV4Tiny(inputs=(416, 416, 3), anchors=anchors, masks=anchor_masks, classes=cfg.num_class,
                           training=True)
            image_shape = tf.keras.layers.Input(shape=(416, 416, 3))
            # model = yolo_body(inputs=image_shape, num_anchors=3, num_classes=cfg.num_class)
            # model = yolo_body(inputs=image_shape, num_anchors=3, num_classes=3, anchors=anchors, masks=anchor_masks,
            #                   training=True)
            model.load_weights(cfg.premodel,by_name=True,skip_mismatch=True)
        # model.backbone.trainable = False
        if cfg.use_CosineDecayRestarts:

            lr_schedule = tf.keras.experimental.CosineDecayRestarts(
                initial_learning_rate=1e-4,

                first_decay_steps=5 * epoch_size,
                t_mul=1.0,
                alpha=1e-2
            )
        else:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-3,
                decay_steps=epoch_size,
                decay_rate=0.92,
                staircase=True
            )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,model= model)
        # checkpoint.restore(tf.train.latest_checkpoint(cfg.checkpoint_dir))
        for epoch in range(500):
            i = 0
            loss = 0
            val_loss = 0
            val_i = 0
            start = time.time()
            for image, label in dataset:
                loss+=distributed_train_step(image,label)
                i+=1
            for image_val, label_val in dataset_val:
                val_loss+=distributed_val_step(image_val,label_val)
                val_i +=1
            end = time.time()

            print("epoch:{} ---->train loss:{:.3f} ---->val loss:{:.3f}----->speed time:{}".format(epoch,loss.numpy()/i,val_loss.numpy() / val_i,end-start))
            # if epoch%10==0:
            #     checkpoint.save(cfg.checkpoint_prefix)
            if val_loss.numpy() / val_i <cfg.best_loss:
                cfg.best_loss = val_loss.numpy() / val_i
                if cfg.yolotiny:
                    model.save_weights('yolov4_tiny_car.h5')
                else:
                    model.save_weights(('yolov4.h5'))


