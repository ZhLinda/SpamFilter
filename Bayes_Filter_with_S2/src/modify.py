#!/usr/bin/env python
# -*- coding:utf-8 -*-
import jieba

fp = open("../data/fixed/train_data_10w.txt","r")
fp2 = open("../data/fixed/ttt.txt","w+")
lines = fp.readlines()
for i in range(2000,2050):
    line = lines[i].strip().split("\t",1)[1].decode("utf-8")
    string = jieba.cut(line,cut_all=False)
    for w in string:
        fp2.write(w.encode("utf-8") + " ")
    fp2.write("\n")
