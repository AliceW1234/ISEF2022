# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:00:42 2022

@author: m255127
"""


import matplotlib.pyplot as plt
import numpy as np
y_test=np.load('/content/drive/MyDrive/testing_data_and_train_data/test_data.npy')
#x_test=np.load('/content/x_test_blur.npy') * 3071
x_test = np.squeeze(x_test)
y_RDN_sharpe_pred=np.squeeze(np.load('/content/y_pred_RDN_sharp_super.npy')) * 3071
y_RDN_pred=np.squeeze(np.load('/content/y_pred_RDN_super.npy')) * 3071
y_unet_pred=np.squeeze(np.load('/content/y_pred_UNet_super.npy')) * 3071
arr=[]
arr.append(x_test[5])
arr.append(y_test[5])
arr.append(y_RDN_sharpe_pred[5])
arr.append(y_RDN_pred[5])
arr.append(y_unet_pred[5])
#arr.append(y_test[0]-y_test[0])

arr.append(x_test[5]-y_test[5])
arr.append(y_test[5]-y_test[5])
arr.append(y_RDN_sharpe_pred[5]-y_test[5])
arr.append(y_RDN_pred[5]-y_test[5])
arr.append(y_unet_pred[5]-y_test[5])

#arr.append(x_test[0]-y_test[0])
m,n=2,5
fid='/content/val_data.npy'
fig, ax = plt.subplots(m, n, tight_layout=True, figsize=(15, 10))
count = 0
titles=['down resolution','Ground truth','RDN sharp rediction', 'RDN prediction', 'U-Net prediction',
        'down resolution-ground truth', 'ground truth-ground truth', 'RDN sharp prediction-ground truth', 'RDN prediction-ground truth', 'U-Net prediction-ground truth']
for i in range(m):
    for j in range(n):
        img = arr[count]
        ax[i, j].imshow(np.rot90(img), cmap='gray',vmin=-160,vmax=240)
        ax[i, j].axis('off')
        ax[i, j].set_title(titles[count], size=6)
        count += 1

plt.tight_layout(pad = 0,  w_pad=0, h_pad=0)
plt.savefig('/content/result_pred.jpeg', dpi=300)
plt.savefig("/work", dpi=300)
#plt.close(fig)

## quantitative comparison
import math
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

def compute_psnr(img1, img2):
    img1 = img1.astype(np.float64) / 1.
    img2 = img2.astype(np.float64) / 1.
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return "Same Image"
    return 10 * math.log10(3071. * 3071. / mse)

psnr1,psnr2=[],[]
ssim1,ssim2=[],[]
rmse1,rmse2=[],[]

y_pred = y_RDN_sharpe_pred

for i in range(x_test.shape[0]):
    psnr1.append(compute_psnr(x_test[i,:,:],y_test[i,:,:]))
    psnr2.append(compute_psnr(y_pred[i],y_test[i,:,:]))

    ssim1.append(ssim(x_test[i,:,:],y_test[i,:,:],win_size=3,data_range=1))
    ssim2.append(ssim(y_pred[i,:,:],y_test[i,:,:],win_size=3,data_range=1))
 
    rmse1.append(np.sqrt(mse(x_test[i,:,:],y_test[i,:,:])))
    rmse2.append(np.sqrt(mse(y_pred[i,:,:],y_test[i,:,:])))
