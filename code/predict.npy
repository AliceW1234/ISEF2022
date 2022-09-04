  # -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:02:31 2022

@author: m255127
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 10:54:35 2022

@author: m255127
"""


# load packages
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import numpy as np
import sys

#sys.path.append('/scratch/wcaodev/ai_patch_classifier/decomp/rdn-ct/utils.py')
#sys.path.append('/scratch/wcaodev/ai_patch_classifier/decomp/rdn-ct/models.py')
#from utils import blur_simu,seg_patch
#from models import RDN
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
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
scale = 2
#scale = 1  # LR缩小的比率
size = 64  # 输入大小，输出的size还要乘以scale
aug_num = 0  # 数据扩充的数量
num_G = 64  # RDN中每个卷积层卷积核的数量
Imgflag = 1  # 图片读取方式:0为RGB图，1为灰度图，2为亮度图中Y图层
if Imgflag == 0:
    channels = 3
else:
    channels = 1
    
input = Input((None, None, channels), name='patch')
#model=RDN(input,num_G,channels)
model=RDN(input,num_G,channels,scale,sr=True)
#model = get_unet(input)
#model = get_unet(input)
model.load_weights('/content/drive/MyDrive/cp_RDN_super.ckpt')
#x_test=np.load('/scratch/wcaodev/ai_patch_classifier/decomp/validation_data/x_test.npy')
# adipose = nib.load('/Pancreas-63502.nii.gz').get_fdata()
# adipose = np.moveaxis(adipose, 2, 0)
# adipose = adipose[0::10]
# #adipose = adipose/adipose.max()

# np.save('/content/adipose.npy', adipose)
adipose = np.load('/content/drive/MyDrive/testing_data_and_train_data/test_data.npy')
#adipose = adipose / 3071
# simulate input/blur data
x_test=np.zeros((len(adipose), 512 , 512))

for i in range(x_test.shape[0]):
    #x_test[i]=blur_simu(adipose[i],blur_sig=1.6,win_size=25,noi_sig=.02)
    #x_test[i]=blur_simu(adipose[i],blur_sig=1.6,win_size=25,noi_sig=.02)
    #temp = resize(np.squeeze(x_test[i]), (x_test[i].shape[0] // 2, x_test.shape[1] // 2),
                       #anti_aliasing=True)
    #x_test[i] = resize(temp, (temp.shape[0] * 2, temp.shape[1] * 2),
                       #anti_aliasing=True) 
    #x_test[i] = down_resolution(adipose[i])
    #x_test[i] = np.expand_dims(down_resolution(adipose[i]),axis=2)
    temp = resize(adipose[i], (adipose[i].shape[0] // 2, adipose[i].shape[1] // 2),
                     anti_aliasing=True)
    x_test[i] = resize(temp, (temp.shape[0] * 2, temp.shape[1] * 2),
                     anti_aliasing=True)
# x_test=nib.load('/content/Pancreas-29673.nii.gz').get_fdata()
#x_test = x_test[0:39321600]
print(x_test.shape)
#np.save('/content/x_test_blur',x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],x_test.shape[2],1))
# print(x_test.shape)
y_pred=np.zeros((len(adipose), 512, 512, 1))
#y_pred = np.moveaxis(y_pred, 1, 0)
# for i in range(x_test.shape[0]):
#     y_pred[i:i+1]=model.predict(x_test[i:i+1])

#np.save('/content/y_pred_RDN_super.npy',y_pred)
