import cv2
import random
import numpy as np

def preprocess(img,imageSize,dataAug=False):

    if img is None:
        img = np.zeros([imageSize[1],imageSize[2]])
    
    if dataAug:
        stretch = (random.random()-0.5)
        wStretched = max(int(img.shape[1] * (1 + stretch)), 1)
        img = cv2.resize(img, (wStretched, img.shape[0]))

    (wt,ht) = imageSize
    (h,w)   = img.shape
    fx = w/wt
    fy = h/ht
    f = max(fx,fy)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
    img = cv2.resize(img,newSize)

    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]]=img

    img = cv2.transpose(target)
    (m, s) = cv2.meanStdDev(img)

    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img

    return img

# img = cv2.imread('data/test.png',cv2.IMREAD_GRAYSCALE)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# img = preprocess(img,(128, 32))
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()