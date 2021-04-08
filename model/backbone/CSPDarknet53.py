import tensorflow as tf
from model.layers import *
class CSPDarkNet(tf.keras.Model):
    def __init__(self,layers):
        super(CSPDarkNet, self).__init__()
        self.inplanes = 32
        self.conv1 = ConvBNActivation(self.inplanes,kernel_size=3)
        self.featrue = [64,128,256,512,1024]
        self.resblock_body1 = ResBlockBody(out_ch=self.featrue[0],num_blocks=layers[0],first=True)
        self.resblock_body2 = ResBlockBody(out_ch=self.featrue[1], num_blocks=layers[1])
        self.resblock_body3 = ResBlockBody(out_ch=self.featrue[2], num_blocks=layers[2])
        self.resblock_body4 = ResBlockBody(out_ch=self.featrue[3], num_blocks=layers[3])
        self.resblock_body5 = ResBlockBody(out_ch=self.featrue[4], num_blocks=layers[4])
    # @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.resblock_body1(x)
        x = self.resblock_body2(x)

        x3 = self.resblock_body3(x)
        x4 = self.resblock_body4(x3)
        x5 = self.resblock_body5(x4)
        return x3,x4,x5
def CSPDarkNet53(pretrained=False,**kwargs):
    model_path = ''
    model = CSPDarkNet([1, 2, 8, 8, 4])
    if pretrained:
        model = tf.saved_model.load(model_path)
    return model


if __name__ == '__main__':
    x = tf.random.uniform((1,416,416,3))
    import time
    s = time.time()
    model = CSPDarkNet53()
    model.build((1,416,416,3))
    model.summary()
    e = time.time()
    print('load model time:{}s'.format(e-s))

    y = model(x,training=True)


    for i in y:
        print(i.shape)


'''
(1, 52, 52, 256)
(1, 26, 26, 512)
(1, 13, 13, 1024)
Total params: 2,257,984
Trainable params: 2,223,872
Non-trainable params: 34,112


Total params: 26,652,512
Trainable params: 26,617,184
Non-trainable params: 35,328

'''


