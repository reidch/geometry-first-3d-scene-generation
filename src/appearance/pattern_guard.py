from __future__ import annotations
import numpy as np
from PIL import Image


def periodicity_score(image_path,mask_path):
    im=np.asarray(Image.open(image_path).convert('L'),np.float32);mask=np.asarray(Image.open(mask_path).convert('L').resize((im.shape[1],im.shape[0]),Image.Resampling.NEAREST))>0
    ys,xs=np.where(mask)
    if len(xs)<512:return 0.0
    crop=im[max(0,ys.min()):ys.max()+1,max(0,xs.min()):xs.max()+1]
    if min(crop.shape)<16:return 0.0
    crop=crop-crop.mean();spec=np.abs(np.fft.fftshift(np.fft.fft2(crop)))
    cy,cx=np.array(spec.shape)//2;spec[max(0,cy-3):cy+4,max(0,cx-3):cx+4]=0
    flat=np.sort(spec.ravel());return float(flat[-12:].sum()/max(flat.sum(),1e-6))
