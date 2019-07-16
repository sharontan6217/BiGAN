#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 12:27:21 2019

@author: sharontan
"""

import biGAN as biGAN
from biGAN import BiGAN



for i in range (5):
    x=biGAN.BiGAN()
    x.predict()
    x.visualising()
    x.clean()
    i+=1
    