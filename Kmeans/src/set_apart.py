#!/usr/bin/env python
# -*- coding:utf-8 -*-

import time

def set_part(ori_data_filename,ori_class_filename,Dl_data_filename,Dl_class_filename,Du_data_filename,Du_class_filename):
    fin = open(ori_data_filename,"r")
    fin_c = open(ori_class_filename,"r")
    fout1 = open(Dl_data_filename,"w")
    fout1_c = open(Dl_class_filename,"w")
    fout2 = open(Du_data_filename,"w")
    fout2_c = open(Du_class_filename,"w")
    lines_data = [line.strip().decode('utf-8') for line in fin.readlines()]
    lines_class = [line.strip().decode('utf-8') for line in fin_c.readlines()]
    size = 17173
    l = 3500
    for i in range(l):
        fout1.write(lines_data[i].encode("utf-8") + "\n")
        fout1_c.write(lines_class[i].encode("utf-8") + "\n")
    for i in range(l,size):
        fout2.write(lines_data[i].encode("utf-8") + "\n")
        fout2_c.write(lines_class[i].encode("utf-8") + "\n")

if __name__ == "__main__":
    ori_data_filename = "../data/derived/train_allwords.txt"
    ori_class_filename = "../data/derived/classLabel.txt"
    Dl_data_filename = "../data/derived/Dl_data.txt"
    Dl_class_filename = "../data/derived/Dl_class.txt"
    Du_data_filename = "../data/derived/Du_data.txt"
    Du_class_filename = "../data/derived/Du_class.txt"
    set_part(ori_data_filename,ori_class_filename,Dl_data_filename,Dl_class_filename,Du_data_filename,Du_class_filename)



