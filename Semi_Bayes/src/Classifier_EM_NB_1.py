#!/usr/bin/env python
# -*- coding:utf-8 -*-

from NaiveBayes import *
import time
import numpy as np

def classifier_init(dictionary,Dl,Cl):
    '''
    :param dictionary: 字典，即特征集
    :param Dl: 标注数据集
    :param Cl: 标注数据集的label
    :return: p_w_spam,p_w_ham,p_spam,p_ham
    '''
    dict_doc_tf_mat = np.zeros((len(dictionary), len(Dl)), np.float64)
    for i in range(len(dictionary)):
        word = dictionary[i]
        for j in range(len(Dl)):
            dict_doc_tf_mat[i][j] = Dl[j].split().count(word)
    #print dict_doc_tf_mat.shape
    #print dict_doc_tf_mat
    is_doc_spam = np.array(Cl)
    #print is_doc_spam.shape
    #print is_doc_spam
    is_doc_ham = 1 - is_doc_spam
    #print dict_doc_tf_mat * is_doc_spam
    up_spam = []
    for i in range(len(dictionary)):
        up_spam.append(sum(dict_doc_tf_mat[i] * is_doc_spam) + 1)

    #print sum_up_spam
    up_ham = []
    for i in range(len(dictionary)):
        up_ham.append(sum(dict_doc_tf_mat[i] * is_doc_ham) + 1)
    down_spam = sum(up_spam)
    down_ham = sum(up_ham)
    p_w_spam = up_spam / down_spam
    p_w_ham = up_ham / down_ham

    #print "up_spam = ",up_spam
    #print "up_ham = ",up_ham
    #print "p_w_spam = ",p_w_spam
    #print "p_w_ham = ",p_w_ham

    numTrainDocs = len(Dl)  # the number of training examples
    p_spam = sum(Cl) / float(numTrainDocs)
    p_ham = 1 - p_spam

    #print p_w_spam,p_w_ham,p_spam,p_ham
    return p_w_spam,p_w_ham,p_spam,p_ham

def E_step(p_w_spam,p_w_ham,p_spam,p_ham,Du,Cu,dictionary):
    '''
    :param p_w_spam: spam类别下出现词典中每个词的概率，list类型
    :param p_w_ham: ham类别下出现词典中每个词的概率，list类型
    :p_spam: spam类型文档所占比例
    :p_ham: ham类型文档所占比例
    :param Du: 未标注数据集
    :param Cu: 未标注数据集的类别
    :dictionary: 字典，即特征集
    :return: p_spam_d,p_ham_d
    '''
    voc = dict(zip(dictionary,range(len(dictionary)))) # for searching quickly
    mul_w_spam = []
    mul_w_ham = []
    min_pr = 0.0000001
    #min_pr = min(min(p_w_ham),min(p_w_spam))
    print "min_pr = ",min_pr
    for item in Du:
        doc = item.strip().split()
        mul_s = 1.0
        mul_h = 1.0
        for word in doc:
            if word in voc:
                #if p_w_spam[voc[word]] != 0:
                    mul_s = mul_s * (p_w_spam[voc[word]])
               # else:
                  #  mul_s = mul_s * min_pr
                #if p_w_ham[voc[word]] != 0:
                    mul_h = mul_h * (p_w_ham[voc[word]])
               # else:
               #     mul_h = mul_h * min_pr
        mul_w_spam.append(mul_s)
        mul_w_ham.append(mul_h)
    up_spam = p_spam * np.array(mul_w_spam) + 1
    up_ham = p_ham * np.array(mul_w_ham) + 1
    down = up_spam + up_ham + len(Dl)
    for i in range(len(down)): #防止除0错误
        if down[i] == 0:
            down[i] = 1
    print "up_spam = ", up_spam
    print "up_ham = ", up_ham
    print "down = ",down
    p_spam_d = up_spam / down
    p_ham_d = up_ham / down

    #print p_spam_d,p_ham_d
    return p_spam_d,p_ham_d


def M_step(p_spam_d,p_ham_d,dictionary,Dl,Cl,Du,Cu):
    p_spam = (sum(Cl) + sum(p_spam_d)) / (len(Dl) + len(Du))
    dict_doc_tf_mat_labeled = np.zeros((len(dictionary), len(Dl)), np.float64)
    for i in range(len(dictionary)):
        word = dictionary[i]
        for j in range(len(Dl)):
            dict_doc_tf_mat_labeled[i][j] = Dl[j].split().count(word)
    dict_doc_tf_mat_unlabeled = np.zeros((len(dictionary), len(Du)), np.float64)
    for i in range(len(dictionary)):
        word = dictionary[i]
        for j in range(len(Du)):
            dict_doc_tf_mat_unlabeled[i][j] = Du[j].split().count(word)

    is_doc_spam = np.array(Cl)
    is_doc_ham = 1 - is_doc_spam

    up_spam = []
    for i in range(len(dictionary)):
        up_spam.append(1 + sum(dict_doc_tf_mat_labeled[i] * is_doc_spam) + sum(dict_doc_tf_mat_unlabeled[i] * p_spam_d))

    up_ham = []
    for i in range(len(dictionary)):
        up_ham.append(1 + sum(dict_doc_tf_mat_labeled[i] * is_doc_ham) + sum(dict_doc_tf_mat_unlabeled[i] * p_ham_d))

    down_spam = sum(up_spam)
    down_ham = sum(up_ham)

    p_w_spam = up_spam / down_spam
    p_w_ham = up_ham / down_ham

    #print p_w_spam, p_w_ham, p_spam, p_ham
    return p_w_spam,p_w_ham,p_spam,p_ham



if __name__ == "__main__":
    Cl = load_class_data('../data/derived/Dl_class_10w.txt')
    Dl = load_train_data('../data/derived/Dl_data_10w.txt')
    Cu = load_class_data('../data/derived/Du_class_10w.txt')
    Du = load_train_data('../data/derived/Du_data_10w.txt')
    fpd = open("../model/dictionary.txt", "r")
    dictionary = [line.strip().decode('utf-8') for line in fpd.readlines()]
    fpd.close()
    p_w_spam, p_w_ham, p_spam, p_ham = classifier_init(dictionary,Dl,Cl)

    loop_count = 2
    for i in range(loop_count):
        p_spam_d,p_ham_d = E_step(p_w_spam,p_w_ham,p_spam,p_ham,Du,Cu,dictionary)

        p_w_spam,p_w_ham,p_spam,p_ham = M_step(p_spam_d,p_ham_d,dictionary,Dl,Cl,Du,Cu)

    p_w_ham = np.log(p_w_ham)
    p_w_spam = np.log(p_w_spam)

    np.savetxt("../p_words_spam.txt", p_w_spam, delimiter="\t")
    np.savetxt("../p_words_ham.txt", p_w_ham, delimiter="\t")

    test_vector = word2vec_tfidf(Du, dictionary)
    classify_result = []
    for vec2classify in test_vector:
        label, t1, t2 = classify(vec2classify, p_w_ham, p_w_spam, p_spam)
        classify_result.append(label)
    evaluate(classify_result, Cu)