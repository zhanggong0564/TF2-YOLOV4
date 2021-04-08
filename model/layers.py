import tensorflow as tf
from tensorflow.keras.layers import *

def Mish():
    def mish(x):
        out = x*tf.nn.tanh(tf.nn.softplus(x))
        return out
    return mish

class ConvBNActivation(tf.keras.Model):
    def __init__(self,ouput_ch,kernel_size,strides=1,activate='mish',bn = True):
        super(ConvBNActivation, self).__init__()
        assert activate in ['relu','leaky','mish']
        self.strides = strides
        if strides==2:
            self.padding = 'valid'
        else:
            self.padding = 'same'
        self.conv = Conv2D(ouput_ch,kernel_size=kernel_size,strides=strides,padding=self.padding,
                           use_bias=not bn,kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                           kernel_initializer=tf.keras.initializers.he_normal())
        if bn:
            self.bn = BatchNormalization()
        if activate=='relu':
            self.activation = ReLU()
        elif activate=='leaky':
            self.activation = LeakyReLU(0.1)
        else:
            self.activation =Mish()
    def call(self, inputs,training = None):
        # tf.print(training)
        if self.strides==2:
            inputs = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        x = self.conv(inputs)
        x = self.bn(x,training)
        x = self.activation(x)
        return x
class ResBlock(Layer):
    def __init__(self,mid_out,fina_out,strides = 1,activate='mish'):
        super(ResBlock, self).__init__()
        self.block = tf.keras.models.Sequential([
            ConvBNActivation(mid_out,kernel_size=1,strides=strides,activate=activate),
            ConvBNActivation(fina_out, kernel_size=3, strides=strides, activate=activate)
        ])
    def call(self, inputs, **kwargs):
        return inputs+self.block(inputs)
class ResBlockBody(tf.keras.Model):
    def __init__(self,out_ch,num_blocks,first =False):
        super(ResBlockBody, self).__init__()
        self.conv1 = ConvBNActivation(out_ch,kernel_size=3,strides=2)
        if first:
            self.route_conv1 = ConvBNActivation(out_ch,kernel_size=1)
            self.route_conv2 = ConvBNActivation(out_ch,kernel_size=1)
            self.blocks = tf.keras.models.Sequential([
                ResBlock(mid_out=out_ch//2,fina_out=out_ch),
                ConvBNActivation(out_ch,kernel_size=1)
            ])
            self.concat_conv=ConvBNActivation(out_ch,1)
        else:
            self.route_conv1 = ConvBNActivation(out_ch//2,kernel_size=1)
            self.route_conv2 = ConvBNActivation(out_ch//2,kernel_size=1)
            conv_list = [ResBlock(mid_out=out_ch//2,fina_out=out_ch//2) for _ in range(num_blocks)]
            conv_list.append(ConvBNActivation(out_ch//2, kernel_size=1))
            self.blocks = tf.keras.models.Sequential(conv_list)
            self.concat_conv=ConvBNActivation(out_ch,1)
    # @tf.function
    def call(self, inputs,training=None):
        x = self.conv1(inputs,training)

        x2 =self.route_conv2(x,training)

        x1 = self.route_conv1(x,training)
        x1 = self.blocks(x1)

        x = concatenate([x1,x2],axis=-1)
        x = self.concat_conv(x)
        return x
class SpatialPyramidPooling(Layer):
    def __init__(self):
        super(SpatialPyramidPooling, self).__init__()
        self.maxpool1 = MaxPooling2D(pool_size=(5, 5),strides=(1,1),padding='same')
        self.maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')
        self.maxpool3 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')
    def call(self, inputs, **kwargs):
        pool1 = self.maxpool1(inputs)
        pool2 = self.maxpool2(inputs)
        pool3 = self.maxpool3(inputs)
        outputs = concatenate([pool1,pool2,pool3,inputs],axis=-1)
        return outputs
class UpSample(Layer):
    def __init__(self,out_ch):
        super(UpSample, self).__init__()
        self.upsample = tf.keras.models.Sequential([
            ConvBNActivation(out_ch,kernel_size=1),
            UpSampling2D(2,interpolation='bilinear')
        ])
    def call(self, inputs, **kwargs):
        x = self.upsample(inputs)
        return x
def make_tfree_conv(filters_list):
    m = tf.keras.models.Sequential([
        ConvBNActivation(filters_list[0],kernel_size=1),
        ConvBNActivation(filters_list[1],kernel_size=3),
        ConvBNActivation(filters_list[0],kernel_size=1)
    ])
    return m
def make_five_conv(filters_list):
    m = tf.keras.models.Sequential([
        ConvBNActivation(filters_list[0], 1),
        ConvBNActivation(filters_list[1], 3),
        ConvBNActivation(filters_list[0], 1),
        ConvBNActivation(filters_list[1], 3),
        ConvBNActivation(filters_list[0], 1),
    ])
    return m
def yolo_head(filters_list):
    m = tf.keras.models.Sequential([
        ConvBNActivation(filters_list[0], 3),
        Conv2D(filters_list[1],1)
    ])
    return m