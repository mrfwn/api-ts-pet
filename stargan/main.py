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

    src = cv2.imread("assets/src.jpeg")
    ref = cv2.imread("assets/ref.jpg")

    t0 = time.time()
    res_img = solver.sample(src,ref)
    t1 = time.time()
    
    cv2.imwrite("teste.jpg",res_img)
    print("Time:")
    print(t1-t0)

if __name__ == '__main__':
    main()
