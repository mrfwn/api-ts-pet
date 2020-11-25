"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import torch

from core.solver import Solver
import cv2
import time


def main():

    solver = Solver()

    with open('imgNames.txt') as f:
        imgName = f.read()

    src = cv2.imread("tmp/uploads/" + imgName)
    ref = cv2.imread("src/stargan/assets/ref.jpg")

    t0 = time.time()
    res_img = solver.sample(src,ref)
    t1 = time.time()
    
    cv2.imwrite("tmp/teste.jpg",res_img)

    with open('output.txt', 'w+') as f:
        f.write('teste.jpg')

if __name__ == '__main__':
    main()
