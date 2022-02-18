#!/usr/bin/env python
# coding=utf-8
'''
Author: luantianyu
LastEditors: Luan Tianyu
email: 1558747541@qq.com
github: https://github.com/tianyuluan/
Date: 2021-09-07 19:26:09
LastEditTime: 2021-12-02 10:35:27
motto: Still water run deep
Description: Modify here please
FilePath: /mmdetection/tools/tt.py
'''
import cv2 
import numpy
feature = x[0].cpu().numpy()
feature_r = feature[0,...]*255
feature_g = feature[13,...]*255
feature_b = feature[34,...]*255
vis_feature = numpy.array([feature_r,feature_g,feature_b])
vis_feature = vis_feature.swapaxes(0,2)
vis_feature = vis_feature.swapaxes(0,1)
cv2.imwrite('/home/lty/桌面/1/原图/4.jpg', vis_feature) 