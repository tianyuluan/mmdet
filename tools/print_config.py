#!/usr/bin/env python
# coding=utf-8
'''
Author: luantianyu
LastEditors: Luan Tianyu
email: 1558747541@qq.com
github: https://github.com/tianyuluan/
Date: 2021-09-13 19:04:07
LastEditTime: 2021-11-30 22:02:10
motto: Still water run deep
Description: Modify here please
FilePath: /mmdetection/tools/print_config.py
'''
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from mmcv import Config, DictAction
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Print the whole config')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg.dump('/home/lty/lty/mmdetection/work_dirs/retinanet.py')
    print(f'Config:\n{cfg.pretty_text}')


if __name__ == '__main__':
    main()
