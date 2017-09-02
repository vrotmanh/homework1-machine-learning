#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 12:38:53 2017

@author: vicenterotman
"""
def displayDigit():
    return

imageNumber = 0
for line in open("data/train.csv"):
    if imageNumber == 0:
        continue
    
    print line
    imageNumber=+1
    break