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

        if pre_result[i] != test_label[i]:
            print i+1

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

def classify(vec2classify, p_words_ham, p_words_spam, p_spam):
    ps = sum(vec2classify * np.log(p_words_spam)) + np.log(p_spam)
    ph = sum(vec2classify * np.log(p_words_ham)) + np.log(1 - p_spam)
    if ps > ph:
        print 1,ps,ph
        return 1,ps,ph
    else:
        print 0,ps,ph
        return 0,ps,ph

def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格 path=path .strip()
    # 去除尾部 \ 符号
    path= path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os. path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print path+ ' 创建成功'
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print path+ ' 目录已存在'
        return False


def save_dictionary(dictionary,filename):
    fw = open(filename,"w")
    for i in range(len(dictionary)):
        fw.write(dictionary[i].encode("utf-8") + "\n")
    fw.flush()
    fw.close()

def load_dictionary(filename):
    fpd = open(filename,"r")
    dictionary = [line.strip().decode('utf-8') for line in fpd.readlines()]
    fpd.close()
    return dictionary

def save_train_vectors(train_vectors,filename):
    print "----------正在保存特征向量----------\n"
    fp = open(filename, "w")  # 存储setOfWords特征向量
    for i in range(len(train_vectors)):
        for j in range(len(train_vectors[0])):
            fp.write(train_vectors[i][j].__str__() + " ")
        fp.write("\n")
    print "----------特征向量保存完成----------\n"

def save_new_vector(vec, filename):
    fp = open(filename, "a+")
    for i in range(len(vec)):
        fp.write(vec[i].__str__() + " ")
    fp.write("\n")

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

def load_predict_results(filename):
    results = []
    fp = open(filename, "r")
    lines = fp.readlines()
    for line in lines:
        results.append(float(line.strip()))
    return results

def load_extra_features(filename,start,len):
    fp = open(filename,"r")
    lines = fp.readlines()
    fea = []
    for i in range(start,start+len):
        #print i
        fea.append(float(lines[i].strip()))
    fea_ret = np.array(fea)
    return fea_ret

def save_mat(mat, filename):
    fp = open(filename, "w")
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            fp.write(mat[i][j].__str__() + " ")
        fp.write("\n")

def load_mat(filename, row, col):
    mat = np.zeros((row, col), np.float64)
    fp = open(filename, "r")
    lines = fp.readlines()
    for i in range(len(lines)):
        line = lines[i].strip().split()
        for j in range(len(line)):
            mat[i][j] = float(line[j].strip())
    return  mat


def save_mat_dict(mat_dict, filename):
    fp = open(filename,"w")
    for item in mat_dict:
        ham = mat_dict[item][0]
        spam = mat_dict[item][1]
        fp.write(item.encode("utf-8") + " " + ham.__str__() + " " + spam.__str__() + "\n")

def load_mat_dict(filename):
    fp = open(filename,"r")
    mat_dict = {}
    lines = fp.readlines()
    for line in lines:
        line = line.strip().split()
        mat_dict.update({line[0].decode("utf-8"):[float(line[1].strip()), float(line[2].strip())]})
    return mat_dict

def count_time(time0):
    time1 = time.time()
    t = time1 - time0
    print "当前用时为： ",t

def save_NB_model(p_words_ham,p_words_spam,p_spam,p_ham,file_prefix):
    fpSpam = open(file_prefix + "/pSpam.txt","w+")
    spam = p_spam.__str__()
    fpSpam.write(spam)
    fpSpam.close()

    fpHam = open(file_prefix + "/pHam.txt", "w+")
    ham = p_ham.__str__()
    fpHam.write(ham)
    fpHam.close()

    np.savetxt(file_prefix + "/p_words_spam.txt",p_words_spam,delimiter="\t")
    np.savetxt(file_prefix + "/p_words_ham.txt",p_words_ham,delimiter="\t")


def load_NB_model(file_prefix):
    p_words_spam = np.loadtxt(file_prefix + "/p_words_spam.txt",delimiter="\t")
    p_words_ham = np.loadtxt(file_prefix + "/p_words_ham.txt",delimiter="\t")
    fr = open(file_prefix + "/pSpam.txt","r")
    pSpam = float(fr.readline().strip())
    fr.close()

    fr = open(file_prefix + "/pHam.txt", "r")
    pHam = float(fr.readline().strip())
    fr.close()

    return p_words_ham,p_words_spam,pSpam,pHam