# load packages
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2  
# sys.path.append('/utils.py')
# sys.path.append('/models.py')
# from utils import blur_simu,seg_patch
# from models import RDN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from imageio import imread, imwrite
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Input, BatchNormalization, Activation, MaxPooling2D, Dropout, Add
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate,Flatten,Dense,Reshape
from tensorflow.keras.callbacks import CSVLogger,LearningRateScheduler,EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import he_normal
import os
import nibabel as nib

# load data
#aapm_dir='/'
#adipose=np.load('/content/Pancreas-00167.nii.gz')
# adipose = nib.load('/content/Pancreas-63502.nii.gz').get_fdata()
# adipose = np.moveaxis(adipose, 2, 0);
# adipose = adipose[0::10]
# simulate input/blur data
# x=np.array([[[0] * 512] * 512] * len(adipose))
# for i in range(x.shape[0]):
#     #x[i]=blur_simu(adipose[i],blur_sig=1.6,win_size=25,noi_sig=.02)
#     #x[i] = cv2.resize(adipose[i], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
#     x[i] = adipose[i]
# print(adipose.max())
# adipose = adipose/adipose.max()
# x = x/x.max()
    
# split data
from sklearn.model_selection import train_test_split

#x_train,x_test,y_train,y_test = train_test_split(x,adipose, test_size=0.15, shuffle=True)
#x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, test_size=0.15, shuffle=True)
#np.save('/x_test.npy',x_test)
#np.save('/y_test.npy',y_test)
# segment patch

#x=np.array([[[0] * 512] * 512] * (len(np.load("/content/train_data_1.npy")) + len(np.load("/content/train_data_2.npy")) + len(np.load("/content/train_data_3.npy")))

x_train = np.concatenate((np.load("/content/drive/MyDrive/testing_data_and_train_data/train_data_1.npy"), np.load("/content/drive/MyDrive/testing_data_and_train_data/train_data_2.npy"), np.load("/content/drive/MyDrive/testing_data_and_train_data/train_data_3.npy"), np.load("/content/drive/MyDrive/testing_data_and_train_data/train_data_4.npy")))
x_val = np.load("/content/drive/MyDrive/testing_data_and_train_data/val_data.npy")

y_train = np.concatenate((np.load("/content/drive/MyDrive/testing_data_and_train_data/train_data_1.npy"), np.load("/content/drive/MyDrive/testing_data_and_train_data/train_data_2.npy"), np.load("/content/drive/MyDrive/testing_data_and_train_data/train_data_3.npy"), np.load("/content/drive/MyDrive/testing_data_and_train_data/train_data_4.npy")))
y_val = np.load("/content/drive/MyDrive/testing_data_and_train_data/val_data.npy")

ttmax = x_train.max()
print(ttmax)

x_train = x_train / ttmax
x_val = x_val / ttmax

y_train = y_train / ttmax
y_val = y_val / ttmax
patchX_train,patchy_train=seg_patch(x_train,y_train,stride=32)
patchX_val,patchy_val=seg_patch(x_val,y_val,stride=32)

# nxtr=np.zeros((len(x_train), 256, 256, 1))
# nxte=np.zeros((len(x_test), 256, 256, 1))
# nxva=np.zeros((len(x_val), 256, 256, 1))
# for i in range(x_train.shape[0]):
#   #nxtr[i] = cv2.resize(x_train[i], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
#   nxtr[i] = np.expand_dims(down_resolution(x_train[i]),axis=2)
# for i in range(x_test.shape[0]):
#   #nxte[i] = cv2.resize(x_test[i], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
#   nxte[i] = np.expand_dims(down_resolution(x_test[i]),axis=2)
# for i in range(x_val.shape[0]):
#   #nxva[i] = cv2.resize(x_val[i], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
#   nxva[i] = np.expand_dims(down_resolution(x_val[i]),axis=2)

npxt = np.zeros((len(patchX_train), 32, 32, 1))
npxv = np.zeros((len(patchX_val), 32, 32, 1))
#print(patchX_train.shape)
for i in range(patchX_train.shape[0]):
  #npxt[i] = blur_simu(patchX_train[i],blur_sig=1.6,win_size=25,noi_sig=.02)
  #npxt[i] = cv2.resize(patchX_train[i], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
  # temp = resize(patchX_train[i], (patchX_train[i].shape[0] // 2, patchX_train[i].shape[1] // 2),
  #                      anti_aliasing=True)
  # #temp = blur_simu(temp,blur_sig=1.6,win_size=25,noi_sig=.02)
  # npxt[i] = resize(temp, (temp.shape[0] * 2, temp.shape[1] * 2),
  #                      anti_aliasing=True)
  npxt[i] = down_resolution(patchX_train[i])
for i in range(patchX_val.shape[0]):
  #npxv[i] = blur_simu(patchX_val[i],blur_sig=1.6,win_size=25,noi_sig=.02)
  #npxv[i] = cv2.resize(patchX_val[i], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
  #npxv[i] = blur_simu(patchX_val[i],blur_sig=1.6,win_size=25,noi_sig=.02)
  npxv[i] = down_resolution(patchX_val[i])
  #npxt[i] = cv2.resize(patchX_train[i], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
  #temp = resize(patchX_val[i], (patchX_val[i].shape[0] // 2, patchX_val[i].shape[1] // 2),
  #                      anti_aliasing=True)
  # temp = blur_simu(temp,blur_sig=1.6,win_size=25,noi_sig=.02)
  #npxv[i] = resize(temp, (temp.shape[0] * 2, temp.shape[1] * 2),
  #t                      anti_aliasing=True)


#npxt = npxt[0:100000];
#npxv = npxv[0:100000];
#patchy_train = patchy_train[0:100000]
#patchy_val = patchy_val[0:100000]

#scale = 1  # LR缩小的比率
scale = 2
size = 64  # 输入大小，输出的size还要乘以scale
aug_num = 0  # 数据扩充的数量
num_G = 64  # RDN中每个卷积层卷积核的数量
Imgflag = 1  # 图片读取方式:0为RGB图，1为灰度图，2为亮度图中Y图层
if Imgflag == 0:
    channels = 3
else:
    channels = 1
    
input = Input((None, None, channels), name='patch')
#model=RDN(input,num_G,channels,scale,sr=False)
model=RDN(input,num_G,channels,scale,sr=True)
#model = get_unet(input)
#model.summary()
#model.load_weights('/cp_super.ckpt')
csv_log = CSVLogger("/content/drive/MyDrive/result_new_RDNsharpen.csv")
checkpoint_path = "/content/drive/MyDrive/cp_new_RDNsharpen.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path) 
# params
warm_up = 1
warm_up_lr = 1e-6
starting_lr = 0.001
max_lr = starting_lr
decay_factor = 0.25
decay_steps = 4
total_epochs = 200
# step decay
epochs_per_step = (total_epochs - warm_up) // decay_steps

def lr_schedule(nepoch):
    if warm_up > 0 and nepoch < warm_up:
        return warm_up_lr
    else:
        lr = max_lr * np.power(
            decay_factor,
            np.floor((nepoch - warm_up) / epochs_per_step))
        return lr
    
def L1_mean_loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))
lrate = LearningRateScheduler(lr_schedule)

model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.999, amsgrad=False)
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1)
              ,loss=L1_mean_loss)
callbacks = [
    EarlyStopping(patience=50, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=6, min_lr=3e-6, verbose=1),lrate, 
    ModelCheckpoint(filepath=checkpoint_path, verbose=0, save_best_only=True, save_weights_only=True),csv_log
]

1# # # Training the Keras model
#history = model.fit(x=[patchX_train], y=[patchy_train], steps_per_epoch=2000,
#                    validation_data=([patchX_val],[patchy_val]), epochs=total_epochs,callbacks=callbacks)
history = model.fit(x=[npxt], y=[patchy_train], steps_per_epoch=2000,
                    validation_data=([npxv],[patchy_val]), epochs=total_epochs,callbacks=callbacks)
