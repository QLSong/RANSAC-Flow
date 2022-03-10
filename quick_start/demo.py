from coarseAlignFeatMatch import CoarseAlign
import sys
sys.path.append('../utils/')
import outil
import cv2
 
sys.path.append('../model/')
import model as model

import PIL.Image as Image 
import os 
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import argparse
import warnings
import torch.nn.functional as F
import pickle 
import pandas as pd
import kornia.geometry as tgm
from itertools import product
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import matplotlib.pyplot as plt 

## composite image    
def get_Avg_Image(Is, It) : 
    
    Is_arr, It_arr = np.array(Is) , np.array(It)
    Imean = Is_arr * 0.5 + It_arr * 0.5
    return Image.fromarray(Imean.astype(np.uint8))


# resumePth = '../model/pretrained/MegaDepth_Theta1_Eta001_Grad1_0.774.pth' ## model for visualization
resumePth = '/workspace/mnt/storage/songqinglong/song/project/RANSAC-Flow/train/MegaDepth_Stage3/BestModel@8_0.765.pth'
kernelSize = 7

Transform = outil.Homography
nbPoint = 4
    

## Loading model
# Define Networks
network = {'netFeatCoarse' : model.FeatureExtractor(), 
           'netCorr'       : model.CorrNeigh(kernelSize),
           'netFlowCoarse' : model.NetFlowCoarse(kernelSize), 
           'netMatch'      : model.NetMatchability(kernelSize),
           }
    

for key in list(network.keys()) : 
    network[key].cuda()
    typeData = torch.cuda.FloatTensor

# loading Network 
param = torch.load(resumePth)
msg = 'Loading pretrained model from {}'.format(resumePth)
print (msg)

for key in list(param.keys()) : 
    network[key].load_state_dict( param[key] ) 
    network[key].eval()

I1 = Image.open('../img/3D_1_1.jpg').convert('RGB')
I2 = Image.open('../img/3D_1_2.jpg').convert('RGB')
# I1 = Image.open('frames/2075.jpg').convert('RGB')
# I2 = Image.open('frames/2050.jpg').convert('RGB')
I1 = Image.fromarray(np.array(I1)[:900, :])
I2 = Image.fromarray(np.array(I2)[:900, :])
plt.figure(figsize=(20, 10))
plt.subplot(1, 3, 1)
plt.imshow(I1)
plt.axis('off')
plt.title('Source Image')
plt.subplot(1, 3, 2)
plt.imshow(I2)
plt.axis('off')
plt.title('Target Image')
plt.subplot(1, 3, 3)
plt.axis('off')
plt.imshow(get_Avg_Image(I1.resize(I2.size), I2))
plt.title('Overlapped Image')
plt.savefig('image1.jpg')

nbScale = 7
coarseIter = 10000
coarsetolerance = 0.05
minSize = 400
imageNet = True # we can also use MOCO feature here
scaleR = 1.2 

coarseModel = CoarseAlign(nbScale, coarseIter, coarsetolerance, 'Homography', minSize, 1, True, imageNet, scaleR)

coarseModel.setSource(I1)
coarseModel.setTarget(I2)

I2w, I2h = coarseModel.It.size
featt = F.normalize(network['netFeatCoarse'](coarseModel.ItTensor))
            
#### -- grid     
gridY = torch.linspace(-1, 1, steps = I2h).view(1, -1, 1, 1).expand(1, I2h,  I2w, 1)
gridX = torch.linspace(-1, 1, steps = I2w).view(1, 1, -1, 1).expand(1, I2h,  I2w, 1)
grid = torch.cat((gridX, gridY), dim=3).cuda() 
warper = tgm.HomographyWarper(I2h,  I2w)

# mask = np.zeros((I2h, I2w))
mask = np.array(I2)[:, :, 0]>100
bestPara, InlierMask = coarseModel.getCoarse(mask)
bestPara = torch.from_numpy(bestPara).unsqueeze(0).cuda()
print(np.sum(InlierMask))
# flowCoarse = warper.warp_grid(bestPara)
flowCoarse = tgm.warp_grid(warper.grid, bestPara)
I1_coarse = F.grid_sample(coarseModel.IsTensor, flowCoarse)
print(flowCoarse.shape, coarseModel.IsTensor.shape, InlierMask.shape)
I1_coarse_pil = transforms.ToPILImage()(I1_coarse.cpu().squeeze())
plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 1)
plt.axis('off')
plt.title('Source Image (Coarse)')
plt.imshow(I1_coarse_pil)
plt.subplot(2, 2, 2)
plt.axis('off')
plt.title('Target Image')
plt.imshow(I2)
plt.subplot(2, 2, 3)
plt.title('Overlapped Image')
plt.imshow(get_Avg_Image(I1_coarse_pil, coarseModel.It))
plt.subplot(2, 2, 4)
plt.title('Match Point Image')
plt.imshow(255*InlierMask)
plt.savefig('image2.jpg')

featsSample = F.normalize(network['netFeatCoarse'](I1_coarse.cuda()))


corr12 = network['netCorr'](featt, featsSample)
flowDown8 = network['netFlowCoarse'](corr12, False) ## output is with dimension B, 2, W, H

flowUp = F.interpolate(flowDown8, size=(grid.size()[1], grid.size()[2]), mode='bilinear')
flowUp = flowUp.permute(0, 2, 3, 1)

flowUp = flowUp + grid

flow12 = F.grid_sample(flowCoarse.permute(0, 3, 1, 2), flowUp).permute(0, 2, 3, 1).contiguous()

I1_fine = F.grid_sample(coarseModel.IsTensor, flow12)
I1_fine_pil = transforms.ToPILImage()(I1_fine.cpu().squeeze())
plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)
plt.axis('off')
plt.title('Source Image (Fine Alignment)')
plt.imshow(I1_fine_pil)
plt.subplot(1, 3, 2)
plt.axis('off')
plt.title('Target Image')
plt.imshow(I2)
plt.subplot(1, 3, 3)
plt.axis('off')
plt.title('Overlapped Image')
plt.imshow(get_Avg_Image(I1_fine_pil, coarseModel.It))
plt.savefig('image3.jpg')