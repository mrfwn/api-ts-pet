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
import random
import string
from random import randrange

def main():

    solver = Solver()

    with open('imgNames.txt') as f:
        imgName = f.read()

    src = cv2.imread("tmp/uploads/" + imgName)
    ref = cv2.imread("src/stargan/assets/ref_"+str(randrange(7))+".jpg")

    res_img = solver.sample(src,ref)
    name_f = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))+'.jpg'
    
    cv2.imwrite("tmp/"+name_f,res_img)

    with open('output.txt', 'w+') as f:
        f.write(name_f)

if __name__ == '__main__':
    main()
