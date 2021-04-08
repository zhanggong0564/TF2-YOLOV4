import os

data_dir = 'train_val.txt'
checkpoint_dir = './tiny_checkpiont'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
best_loss = 20
num_class = 3
Bacthsize = 6*2
val_split = 0.3
use_mosico = True
yolotiny = True
label_smoothing = 0.01
premodel = 'coco_voc_model/yolov4_tiny_weights_coco.h5'
pred_image = True
use_CosineDecayRestarts = True
