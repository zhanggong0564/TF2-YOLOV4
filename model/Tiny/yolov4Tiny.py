from utils__.utils import yolo_eval,yolo_head
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np

def DarknetConv(x,filters,size,strides = 1,batch_norm = True,dilation_rate = 1):
    if strides ==1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1,0),(1,0)))(x)
        padding = 'valid'
    x = Conv2D(filters=filters,kernel_size=size,strides= strides,padding=padding,use_bias=not batch_norm,dilation_rate=dilation_rate)(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x

def route_group(input_layer, groups, group_id):
    # 对通道数进行均等分割，我们取第二部分
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]
def resblock_body(x,num_filters):
    x = DarknetConv(x,num_filters,3)
    route = x
    x = Lambda(route_group,arguments={'groups':2,'group_id':1})(x)
    x = DarknetConv(x,num_filters//2,3)

    route_1 = x
    x = DarknetConv(x,num_filters//2,3)
    x = concatenate([x,route_1])

    x = DarknetConv(x,num_filters,1)
    feat = x

    x = concatenate([route,x])
    x = MaxPool2D(pool_size=[2,2])(x)

    return x,feat

def ASPP(x,out_ch,os):
    if os==16:
        dilation = [1,6,12,18]
    else:
        dilation = [1,12,24,36]
    aspp1 = DarknetConv(x,out_ch,size=1,dilation_rate=dilation[0])
    aspp2 = DarknetConv(x,out_ch,size=3,dilation_rate=dilation[1])
    aspp3 = DarknetConv(x,out_ch,size=3,dilation_rate=dilation[2])
    aspp4 = DarknetConv(x,out_ch,size=3,dilation_rate=dilation[3])
    x = GlobalAveragePooling2D()(x)
    x = tf.expand_dims(x,axis=1)
    x = tf.expand_dims(x,axis = 1)
    gp = tf.keras.models.Sequential([
        Conv2D(256,kernel_size=1,strides=1,use_bias=False),
        BatchNormalization(),
        ReLU()
    ])(x)
    gp = tf.keras.layers.UpSampling2D(13,interpolation='bilinear')(gp)
    x = Concatenate()([aspp1,aspp2,aspp3,aspp4,gp])
    x = Conv2D(256,kernel_size=1,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
def Darknet_tiny(x,name = None):
    x = DarknetConv(x,32,3,strides=2)
    x = DarknetConv(x,64,3,strides=2)
    x,_ = resblock_body(x,num_filters=64)
    x,_ = resblock_body(x,num_filters=128)
    x,feat1 = resblock_body(x,num_filters=256)
    x = DarknetConv(x,512,3)
    # x = ASPP(x,512,os=16)
    feat2 = x
    return feat1,feat2

def YoloV4Tiny(inputs,anchors,masks,classes,training,name=None):
    inputs = Input(inputs,name='input')
    feat1,feat2 = Darknet_tiny(inputs)

    p5 = DarknetConv(feat2,256,1)
    p5_out = DarknetConv(p5,512,3)
    p5_out = Conv2D(len(masks[0])*(classes+5),kernel_size=1,padding='same')(p5_out)


    p5_upsample = DarknetConv(p5,128,1)
    p5_upsample = UpSampling2D(2)(p5_upsample)

    p4 = Concatenate()([feat1,p5_upsample])
    p4_output = DarknetConv(p4,256,3)
    #######注意分类层的定义,当分类层定义了bn,relu,loss很难收敛
    p4_output = Conv2D(len(masks[0])*(classes+5),kernel_size=1,padding='same')(p4_output)
    if training:
        return tf.keras.Model(inputs,[p5_out,p4_output],name=name)
    else:
        outputs = Lambda(lambda x: yolo_eval(x, anchors, classes),
                         name='yolo_nms')([p5_out, p4_output])
        return tf.keras.Model(inputs, outputs, name='yolov4_tiny')



if __name__ == '__main__':
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
    yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                             (59, 119), (116, 90), (156, 198), (373, 326)],
                            np.float32)
    yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    x = tf.random.uniform((1, 416, 416, 3))
    # model = Yolov4(4,yolo_anchors,yolo_anchor_masks)
    model = YoloV4Tiny(inputs=(416, 416, 3), anchors=yolo_anchors, masks=yolo_anchor_masks, classes=4, training=False)
    model.summary()
    # model.load_weights('/home/zhanggong/Downloads/yolov4_tiny_weights_voc.h5',by_name=True,skip_mismatch=True)
    y = model(x)
    for i in y:
        print(i.shape)
    # x = tf.random.uniform((1, 13, 13, 256))
    # model = ASPP(x,128,os=16)


