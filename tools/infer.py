#!/usr/bin/env python
# coding=utf-8
'''
Author: luantianyu
LastEditors: Luan Tianyu
email: 1558747541@qq.com
github: https://github.com/tianyuluan/
Date: 2021-09-29 11:51:36
LastEditTime: 2021-12-01 20:20:10
motto: Still water run deep
Description: Modify here please
FilePath: /mmdetection/tools/infer.py
'''
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
config_file = '/home/lty/lty/mmdetection/work_dirs/centernet2_cascade_res50_fpn_4x_coco/centernet2_cascade_res50_fpn_4x_coco.py'
checkpoint_file = '/home/lty/lty/mmdetection/work_dirs/centernet2_cascade_res50_fpn_4x_coco/epoch_24.pth'
image = '/home/lty/桌面/1/原图/b265b9cf-2d517da6.jpg'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
results = inference_detector(model, image)
show_result_pyplot(model, image, results)