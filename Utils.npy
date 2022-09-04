# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 16:46:47 2022

@author: m255127
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 16:46:47 2022

@author: m255127
"""

from skimage.filters import gaussian
import numpy as np
from skimage.transform import resize

def down_resolution(img,rate=(2,2)):
    h,w=rate
    return resize(img,(img.shape[0]//h,img.shape[1]//w),anti_aliasing=False)

def blur_simu(img,blur_sig=1.6,win_size=25,noi_sig=2):
    t = (((win_size - 1)/2)-0.5)/blur_sig
    im_blur = gaussian(img, sigma=blur_sig, truncate=t)
    noise = np.random.normal(0,noi_sig,img.shape)
    im_blur_noi = im_blur+noise
    return im_blur_noi

def seg_patch(x,y,stride=64,channels = 1):    
    s,h,w,c=x.shape[0],x.shape[1],x.shape[2],channels
    x=np.reshape(x,(s,h,w,c))
    s,h,w,c=y.shape[0],y.shape[1],y.shape[2],channels
    y=np.reshape(y,(s,h,w,c))
    patch1_train=[]
    patch2_train=[]
    
    for i in range(s):
        img = x[i,:,:,:]
        img3 = y[i,:,:,:]
        for j in range(0,(h+1-64),stride):
            for k in range(0,(w+1-64),stride):
                t = img[j:j+64,k:k+64,:]
                t3 = img3[j:j+64,k:k+64,:] 
                if np.sum(t.flatten())!= 0:
                    patch1_train.append(t)
                    patch2_train.append(t3)         
    x_patch = np.asarray(patch1_train)
    y_patch = np.asarray(patch2_train)    
    return x_patch,y_patch
