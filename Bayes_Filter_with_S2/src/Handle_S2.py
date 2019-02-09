#!/usr/bin/env python
# -*- coding:utf-8 -*-
from numpy import *
from sklearn.naive_bayes import GaussianNB
import numpy as np
from NaiveBayes import *
import gc
from sklearn import preprocessing

def load_extra_features(filename,start,len):
    fp = open(filename,"r")
    lines = fp.readlines()
    fea = []
    for i in range(start,start+len):
        #print i
        fea.append(float(lines[i].strip()))
    fea_ret = np.array(fea)
    return fea_ret

def save_train_vectors(train_vectors,filename):
    print "----------正在保存特征向量----------\n"
    fp = open(filename, "w")  # 存储setOfWords特征向量
    for i in range(len(train_vectors)):
        for j in range(len(train_vectors[0])):
            fp.write(train_vectors[i][j].__str__() + " ")
        fp.write("\n")
    print "----------特征向量保存完成----------\n"

def Handle_S2(train_data,train_label):
    l = len(train_data)
    train_vectors_S2 = np.zeros((l, 5), np.float64)

    fea_1 = load_extra_features("../model/ext_feature_1.txt", 0, l)
    #fea_2 = load_extra_features("../model/ext_feature_2.txt", 0, l)
    fea_3 = load_extra_features("../model/ext_feature_3.txt", 0, l)
    #fea_4 = load_extra_features("../model/ext_feature_4.txt", 0, l)
    #fea_5 = load_extra_features("../model/ext_feature_5.txt", 0, l)
    fea_6 = load_extra_features("../model/ext_feature_6.txt", 0, l)
    fea_7 = load_extra_features("../model/ext_feature_7.txt", 0, l)
    fea_8 = load_extra_features("../model/ext_feature_8.txt", 0, l)

    for i in range(l):
        train_vectors_S2[i][0] = fea_1[i]
        #train_vectors_S2[i][1] = fea_2[i]
        train_vectors_S2[i][1] = fea_3[i]
        #train_vectors_S2[i][3] = fea_4[i]
        #train_vectors_S2[i][4] = fea_5[i]
        train_vectors_S2[i][2] = fea_6[i]
        train_vectors_S2[i][3] = fea_7[i]
        train_vectors_S2[i][4] = fea_8[i]

    #del fea_1, fea_2, fea_3, fea_4, fea_5, fea_6, fea_7, fea_8
    gc.collect()

    min_max_scaler = preprocessing.MinMaxScaler()  # 将数据按列缩放到[0,1]之间
    train_vectors_scaled_S2 = min_max_scaler.fit_transform(train_vectors_S2)
    save_train_vectors(train_vectors_scaled_S2, "../model/train_vectors_S2.txt")

    clf = GaussianNB()
    clf.fit(train_vectors_scaled_S2,train_label)
    print clf.class_count_
    print clf.class_prior_
    print clf.classes_
    print clf.priors
    print clf.theta_ # mean of each feature per class
    print clf.sigma_ # variance of each feature per class
    return clf.theta_,clf.sigma_,min_max_scaler


