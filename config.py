import os

data_dir = 'data_info/2007_train_fruit.txt'
checkpoint_dir = './tiny_checkpiont'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
best_loss = 20
num_class = 4
Bacthsize = 8
val_split = 0.3
use_mosico = True
yolotiny = False
label_smoothing = 0.01
premodel = 'yolov4_tiny.h5'
pred_image = True
use_CosineDecayRestarts = True
