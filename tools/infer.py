#!/usr/bin/env python
# coding=utf-8
'''
Author: luantianyu
LastEditors: Luan Tianyu
email: 1558747541@qq.com
github: https://github.com/tianyuluan/
Date: 2021-09-29 11:51:36
LastEditTime: 2022-02-18 16:20:43
motto: Still water run deep
Description: Modify here please
FilePath: /mmdet/tools/infer.py
'''
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
config_file = '/home/lty/lty/mmdet/work_dirs/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco.py'
checkpoint_file = '/home/lty/lty/mmdet/work_dirs/yolox_l_8x8_300e_coco/latest.pth'
image = '/home/lty/图片/b6c695f0f1964d8f8aa352b3d6d8f0e3.jpeg'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
results = inference_detector(model, image)
show_result_pyplot(model, image, results)