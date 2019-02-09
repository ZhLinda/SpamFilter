#!/usr/bin/env python
# -*- coding:utf-8 -*-
import random

if __name__ == "__main__":
    fin = open("../data/derived/pirate90000001_unknown","r")
    fout = open("../data/to_be_labeled_4w.txt","a+")
    lines = fin.readlines()
    len = len(lines)
    random_nums = [random.randint(0,len) for _ in range(2000)]
    print random_nums
    for i in random_nums:
        fout.write(lines[i])