from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob
import sys
import argparse

from PIL import Image

if __name__=='__main__':

    parser= argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='')
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--dst', type=str, default='')
    args= parser.parse_args()

    if not os.path.exists(args.dst):
        os.mkdir(args.dst)

    files= glob.glob(os.path.join(args.src, '*'));
    for file in files:
        dir, name= os.path.split(file)
        base, ext = os.path.splitext(name)
        if not ext in ['.jpg', '.png', '.gif']:
            continue
        img = Image.open(file)
        if img.mode!= 'RGB':
            img= img.convert('RGB')
        img_resized = img.resize((args.width, args.height))
        img_resized.save(os.path.join(args.dst, name));
