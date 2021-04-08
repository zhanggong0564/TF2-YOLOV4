from model.layers import *
from model.backbone.CSPDarknet53 import CSPDarkNet53

class PANet(tf.keras.Model):
    def __init__(self,final_out3):
        super(PANet, self).__init__()
        self.conv1 = make_tfree_conv([512,1024])
        self.spp = SpatialPyramidPooling()
        self.conv2 = make_tfree_conv([512,1024])

        self.upsample1 = UpSample(256)
        self.conv_for_p4 = Conv2D(256,kernel_size=1,padding='same')
        self.conv3 = make_five_conv([256,512])

        self.upsample2 = UpSample(128)
        self.conv_for_p3 = Conv2D(128, kernel_size=1, padding='same')
        self.conv4 = make_five_conv([128, 256])

        self.yolohead3 = yolo_head([256,final_out3])

        self.down_sample1 = ConvBNActivation(256,kernel_size=3,strides=2)
        self.conv5 = make_five_conv([256,512])
        self.yolohead2 = yolo_head([512, final_out3])

        self.down_sample2 = ConvBNActivation(512, kernel_size=3, strides=2)
        self.conv6 = make_five_conv([512,1024])

        self.yolohead1 = yolo_head([512, final_out3])
    # @tf.function
    def call(self, inputs,training=None):
       x3,x2,x1 = inputs
       p5 = self.conv1(x1)
       p5 = self.spp(p5)
       p5 = self.conv2(p5)

       p5_upsample = self.upsample1(p5)

       x2 = self.conv_for_p4(x2)
       x2 = concatenate([x2,p5_upsample],axis=-1)
       x2 = self.conv3(x2)

       p4_upsameple = self.upsample2(x2)

       x3 = self.conv_for_p3(x3)
       x3 = concatenate([x3,p4_upsameple],axis=-1)
       x3 =self.conv4(x3)


       downsample1 = self.down_sample1(x3)
       x2 = concatenate([x2,downsample1],axis=-1)
       x2 = self.conv5(x2)

       downsample2 = self.down_sample2(x2)
       x1 = concatenate([p5,downsample2],axis=-1)
       x1 = self.conv6(x1)

       out3 = self.yolohead3(x3)
       out2 = self.yolohead2(x2)
       out1 = self.yolohead1(x1)
       return out1,out2,out3









