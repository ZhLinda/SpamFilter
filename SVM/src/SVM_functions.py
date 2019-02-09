#!/usr/bin/env python
# -*- coding:utf-8 -*-

import time
from sklearn import preprocessing
import numpy as np

def evaluate(pre_result, test_label):
    TP = 0; FN = 0; FP = 0; TN = 0
    length = len(pre_result)
    print pre_result
    print test_label
    for i in range(length):
        if pre_result[i] == 1 and test_label[i] == 1:
            TP += 1
        elif pre_result[i] == 1 and test_label[i] == 0:
            FP += 1
        elif pre_result[i] == 0 and test_label[i] == 1:
            FN += 1
        else:
            TN += 1
    print TP,FN,FP,TN
    Precise = float(TP) / float(TP + FP)
    Recall = float(TP) / float(TP + FN)
    F1 = float(2 * TP) / float(2 * TP + FP + FN)
    print "精确率为： ",Precise
    print "召回率为： ",Recall
    print "F1值为： ",F1

def save_dictionary(dictionary,filename):
    print "----------正在保存字典----------\n"
    fw = open(filename,"w")
    for i in range(len(dictionary)):
        fw.write(dictionary[i].encode("utf-8") + "\n")
    fw.flush()
    fw.close()
    print "----------字典保存完成----------\n"

def load_dictionary(filename):
    print "----------正在加载字典----------\n"
    fpd = open(filename,"r")
    dictionary = [line.strip().decode('utf-8') for line in fpd.readlines()]
    fpd.close()
    print "----------字典加载完成----------\n"
    return dictionary

def save_train_vectors(train_vectors,filename):
    print "----------正在保存特征向量----------\n"
    fp = open(filename, "w")  # 存储setOfWords特征向量
    for i in range(len(train_vectors)):
        for j in range(len(train_vectors[0])):
            fp.write(train_vectors[i][j].__str__() + " ")
        fp.write("\n")
    print "----------特征向量保存完成----------\n"

def load_train_vectors(len_doc,len_dictionary,filename):
    print "----------正在加载特征向量----------\n"
    train_vectors = np.zeros((len_doc, len_dictionary), np.float64)
    fp = open(filename, "r")
    lines = fp.readlines()
    l = len(lines)
    for i in range(l):
        line = lines[i].strip().split()
        size = len(line)
        for j in range(size):
            train_vectors[i][j] = float(line[j])
    print "----------特征向量加载完成----------\n"
    return train_vectors

def load_extra_features(filename,start,len):
    fp = open(filename,"r")
    lines = fp.readlines()
    fea = []
    for i in range(start,start+len):
        #print i
        fea.append(float(lines[i].strip()))
    fea_ret = np.array(fea)
    return fea_ret