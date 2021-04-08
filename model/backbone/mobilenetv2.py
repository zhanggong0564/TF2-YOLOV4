import tensorflow as tf



# basemodel = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,weights='imagenet',input_shape=(416,416,3))



# basemodel.summary()
def get_premodel():
    basemodel = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet',
                                                               input_shape=(416, 416, 3))
    outputs = [
        basemodel.get_layer('block_6_expand_relu').output,
        basemodel.get_layer('block_13_expand_relu').output,
        basemodel.get_layer('out_relu').output
    ]
    model = tf.keras.Model([basemodel.inputs],outputs)
    return model
class Mobilnetv2(tf.keras.Model):
    def __init__(self):
        super(Mobilnetv2, self).__init__()
        self.basemodel = get_premodel()
        self.conv1_256 = tf.keras.layers.Conv2D(256,kernel_size=1)
        self.conv1_512 = tf.keras.layers.Conv2D(512,kernel_size=1)
        self.conv1_1024 = tf.keras.layers.Conv2D(1024,kernel_size=1)
    def call(self, inputs):
        x1,x2,x3 = self.basemodel(inputs)
        out1 = self.conv1_256(x1)
        out2= self.conv1_512(x2)
        out3 = self.conv1_1024(x3)
        return out1,out2,out3

if __name__ == '__main__':
    # model = get_premodel()
    model = Mobilnetv2()
    x = tf.random.normal((1,416,416,3))
    y = model(x)
    for i in y:
        print(i.shape)