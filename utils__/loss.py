from utils__.utils import yolo_boxes,box_ciou,broadcast_iou,_smooth_labels
import tensorflow as tf
def YoloLoss(anchors,classes = 1,ignore_thresh= 0.5,label_smoothing = False):
    '''

    :param anchors: 大中小的anchors中的一个
    :param classes: 类别数目
    :param ignore_thresh:
    :return:
    '''
    def yolo_loss(y_true,y_pred):
        '''

        :param y_true:n,13,13,3,(x,y,w,h,conf,c)->5+n_class
        :param y_pred:n,13,13,3*(5+n_class)
        :return:
        '''
        #处理网络的输出pre
        pred, _, _,_,pred_xywh = yolo_boxes(y_pred,anchors,classes)
        objectness, class_probs = pred[...,4:5],pred[...,5:]


        true_box, object_mask, true_class_idx = tf.split(
            y_true, (4, 1, classes), axis=-1)
        # tf.print(object_mask[object_mask>0.5])
        if label_smoothing:
            true_class_idx = _smooth_labels(true_class_idx, label_smoothing)

        box_loss_scale = 2 - true_box[..., 2:3] * true_box[..., 3:4]
        ciou = box_ciou(pred_xywh, true_box)

        #ciou loss
        ciou_loss = object_mask * box_loss_scale * (1 - ciou)

        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_xywh, true_box, object_mask[...,0]),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)
        ignore_mask = tf.expand_dims(ignore_mask,-1)
        _confidence_loss = tf.expand_dims(tf.keras.losses.binary_crossentropy(object_mask, objectness, from_logits=True),-1)
        confidence_loss = object_mask * _confidence_loss + \
                          (1 - object_mask) * _confidence_loss * ignore_mask
        class_loss = object_mask * tf.expand_dims(tf.keras.losses.categorical_crossentropy(true_class_idx, class_probs, from_logits=True), -1)

        location_loss = tf.reduce_sum(ciou_loss,axis=[1,2,3])
        # tf.print(location_loss)
        confidence_loss = tf.reduce_sum(confidence_loss,axis=[1,2,3])
        # tf.print('confloss',confidence_loss)
        class_loss = tf.reduce_sum(class_loss,axis=[1,2,3])
        # tf.print(class_loss)
        total = 2*location_loss+confidence_loss+class_loss
        return total
    return yolo_loss
if __name__ == '__main__':
    import numpy as np

    yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                             (59, 119), (116, 90), (156, 198), (373, 326)],
                            np.float32)
    yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    yolo_loss = YoloLoss(yolo_anchors[yolo_anchor_masks[0]],classes=1)
    pred = tf.ones((2, 13, 13, 3*(1 + 5)))
    true = tf.ones((2, 13, 13, 3,5+1))
    a = [[0., 0., 0., 1.],
         [0., 0., 0., 1.],
         [0., 0., 0., 1.]
         ]
    b = [[0.,0.,0.,1.],
         [0., 0., 0., 1.],
         [0., 0., 0., 1.]
         ]
    a = tf.constant(a)
    b = tf.constant(b)
    v = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(a,b)
    print(v)
    print(yolo_loss(true,pred))
    '''
    array([[362.68976],
       [318.35895]], dtype=float32)>, <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[1708.0424],
       [1576.0569]], dtype=float32)>, <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[6023.1206],
       [5754.828 ]], dtype=float32)>]
    
    '''