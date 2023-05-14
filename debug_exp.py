import cv2
import os
import numpy as np

# img1 = cv2.imread('./picture/000002.jpg')
# img2 = cv2.imread('./picture/000002.png')
# res = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
# # imgadd = cv2.add(img1,img2,mask= True)
# cv2.imshow('imgadd', res)
# cv2.waitKey(0)
from dataset.gui_read_images import get_tensor_images

a = get_tensor_images(r'F:\graproj\data\JPEGImages\bus_010')
h,w = a[0].shape[1:]
print(a[0].shape)
print(h,w)
print(a)