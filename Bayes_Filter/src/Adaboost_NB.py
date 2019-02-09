#!/usr/bin/env python
# -*- coding:utf-8 -*-

from NaiveBayes import *
import random
import numpy as np
from Feature_extraction import *


def train_AdaBoost_NB(traindata, testdata, trainlabel, iterateNum = 100):

    method = ['IG','CHI','MI']
    m = method[0]


    flag = 1
    if flag == 1: #重新选择特征，训练模型
        print "----------正在进行特征选择----------"
        feature_set = feature_selection(traindata, trainlabel, m)
        print len(feature_set)
        print "----------特征选择完成----------\n"
        # size = 0.1 * len(feature_set)
        size = 3000
        dictionary = feature_set[:int(size)]  # 选取前1000个词词作为特征
        # for term in dictionary:
        # print term

        print "----------正在进行word-to-vector转化----------"
        train_vectors = word2vec_setOfWords(traindata, dictionary)
        print "----------word-to-vector转化完成----------\n"

        print "----------正在进行模型训练----------"
        p_words_ham, p_words_spam, p_spam = trainNB(train_vectors, trainlabel)
        save_NB_model(p_words_ham, p_words_spam, p_spam, dictionary)

        print "----------模型训练完成----------\n"
        del feature_set,train_vectors

    else: #载入储存的模型
        p_words_ham,p_words_spam,p_spam,dictionary = load_NB_model()

    # 交叉验证
    test_words = []
    test_words_class = []

    # 采用boosting的方法进行抽样
    test_count = 90000
    test_words = traindata
    test_words_class = trainlabel
    '''
    for i in range(test_count):
        randomIndex = int(random.uniform(0, len(corpus)))
        test_words_class.append(class_label[randomIndex])
        test_words.append(corpus[randomIndex])
        del (corpus[randomIndex])
        del (class_label[randomIndex])'''

    BOOST = np.ones(len(dictionary)) #调整因子BOOST，其作用是调整词汇表中某一词汇的“垃圾程度”

    boost_errorRate = {}
    minErrorRate = np.inf
    for i in range(iterateNum):
        errorCount = 0.0
        vec2test = word2vec_setOfWords(test_words, dictionary)
        for j in range(test_count):
            pred_label,ps,ph = classify_AB(vec2test[j],p_words_ham,p_words_spam,p_spam,BOOST)

            if pred_label != test_words_class[j]:
                errorCount += 1
                alpha = ps - ph
                if alpha > 0: #样本是ham,被预测成spam
                    # BOOST[样本包含的词汇] = np.abs(BOOST[样本包含的词汇] - np.exp(alpha) / BOOST[样本包含的词汇])
                    BOOST[vec2test[j] != 0] = np.abs((BOOST[vec2test[j] != 0] - np.exp(alpha)) / BOOST[vec2test[j] != 0])
                else: #样本是spam,被预测成ham
                    # BOOST[样本包含的词汇] = BOOST[样本包含的词汇] + np.exp(alpha) / BOOST[样本包含的词汇]
                    BOOST[vec2test[j] != 0] = (BOOST[vec2test[j] != 0] + np.exp(alpha)) / BOOST[vec2test[j] != 0]
        print "BOOST:",BOOST
        errorRate = errorCount / test_count
        if errorRate < minErrorRate:
            minErrorRate = errorRate
            boost_errorRate['minErrorRate'] = minErrorRate
            boost_errorRate['BOOST'] = BOOST
        print "第 %d 轮迭代，错误个数 %d，错误率 %f" % (i,errorCount,errorRate)
        if errorRate == 0.0:
            break
    boost_errorRate['dictionary'] = dictionary
    boost_errorRate['p_words_spam'] = p_words_spam
    boost_errorRate['p_words_ham'] = p_words_ham
    boost_errorRate['p_spam'] = p_spam

    test_vector = word2vec_setOfWords(testdata, dictionary)
    classify_result = []
    for vec2classify in test_vector:
        label,t1,t2 = classify_AB(vec2classify,p_words_ham,p_words_spam,p_spam,BOOST)
        classify_result.append(label)

    return boost_errorRate,classify_result

def classify_AB(vec2classify, p_words_ham, p_words_spam, p_spam,BOOST):
    ps = sum(vec2classify * p_words_spam * BOOST) + np.log(p_spam)
    ph = sum(vec2classify * p_words_ham) + np.log(1 - p_spam)
    if ps > ph:
        return 1,ps,ph
    else:
        return 0,ps,ph

def save_AdaBoostNB_model(boost_errorRate):
    np.savetxt("../model/AB_p_words_spam.txt", boost_errorRate['p_words_spam'], delimiter="\t")
    np.savetxt("../model/AB_p_words_ham.txt", boost_errorRate['p_words_ham'], delimiter="\t")
    np.savetxt("../model/AB_pSpam.txt",np.array([boost_errorRate['p_spam']]), delimiter="\t")
    np.savetxt("../model/AB_train_BOOST.txt",boost_errorRate['BOOST'], delimiter="\t")
    np.savetxt("../model/AB_min_error_rate",np.array([boost_errorRate['minErrorRate']]), delimiter="\t")
    dictionary = boost_errorRate['dictionary']
    fw = open("AB_dictionary.txt","w")
    for i in range(len(dictionary)):
        fw.write(dictionary[i].encode("utf-8") + "\n")
    fw.flush()
    fw.close()

def load_AdaBoostNB_model():
    train_BOOST = np.loadtxt("../model/AB_train_BOOST.txt", delimiter="\t")
    train_min_error_rate = np.loadtxt("../model/AB_min_error_rate.txt", delimiter="\t")
    p_words_spam = np.loadtxt("../model/AB_p_words_spam.txt", delimiter="\t")
    p_words_ham = np.loadtxt("../model/AB_p_words_ham.txt", delimiter="\t")
    p_spam = np.loadtxt("../model/AB_pSpam.txt", delimiter="\t")
    fpd = open("../model/AB_dictionary.txt", "r")
    dictionary = [line.strip().decode('utf-8') for line in fpd.readlines()]
    fpd.close()
    return dictionary,train_BOOST,train_min_error_rate,p_words_spam,p_words_ham,p_spam

