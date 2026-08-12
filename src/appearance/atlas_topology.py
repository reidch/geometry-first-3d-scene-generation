from __future__ import annotations
from collections import deque
import numpy as np
from PIL import Image,ImageFilter


def close_reachable(mask, radius=2):
    im=Image.fromarray((np.asarray(mask,bool)*255).astype(np.uint8),'L')
    if radius>0:
        im=im.filter(ImageFilter.MaxFilter(radius*2+1)).filter(ImageFilter.MinFilter(radius*2+1))
    return np.asarray(im)>0


def connected_labels(mask):
    mask=np.asarray(mask,bool);h,w=mask.shape;labels=np.zeros((h,w),np.int32);label=0
    for y,x in zip(*np.where(mask & (labels==0))):
        if labels[y,x]!=0:continue
        label+=1;labels[y,x]=label;q=deque([(y,x)])
        while q:
            cy,cx=q.popleft()
            for ny,nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if 0<=ny<h and 0<=nx<w and mask[ny,nx] and labels[ny,nx]==0:
                    labels[ny,nx]=label;q.append((ny,nx))
    return labels,label
