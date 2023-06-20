#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/2/22 20:19
@File    : voc上色.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import os.path as osp
import re
import subprocess
import sys

from tqdm import tqdm

sys.path = ['.', './src'] + sys.path  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', type=str)
parser.add_argument('-t', '--target', type=str)
args = parser.parse_args()

os.makedirs(args.target, exist_ok=True)

for label_file_name in tqdm(os.listdir(args.source), dynamic_ncols=True):
    if not label_file_name.endswith('.png'):
        continue

    clean_label_file_name = re.search(r'(\d+)\.png', label_file_name).group(1) + '.png'

    subprocess.run(['mv', osp.join(args.source, label_file_name), osp.join(args.target, clean_label_file_name)])
