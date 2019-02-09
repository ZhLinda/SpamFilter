#!/usr/bin/env python
# -*- coding:utf-8 -*-

from NaiveBayes import *
import time
from Adaboost_NB import *

if __name__ == "__main__":

    time0 = time.time()

    print "----------正在加载停用词表----------"
    #stopWords = load_stop_words()  # get the stopwords list
    print "----------停用词表加载完成----------\n"

    print "----------正在进行文本分词----------"
    #get_data_segmented(stopWords)  # 使用JIEBA分词对所有文本进行分词
    #del stopWords
    #gc.collect()
    print "------------文本分词完成------------\n"

    print "----------正在进行文本bigram切分----------"
    #countNgram(2)
    print "------------文本bigram切分完成------------\n"

    print "----------正在合并分词结果----------"
    #merge_words()
    print "----------合并分词结果完成----------\n"

    print "----------正在加载类别信息----------"
    class_label = load_class_data('../data/derived/classLabel_10w.txt')
    print "----------类别信息加载完成----------\n"

    print "----------正在加载语料库----------"
    train_corpus = load_train_data('../data/derived/train_allwords_10w.txt')
    print "----------语料库加载完成----------\n"

    # 交叉验证，验证集占总训练集的10%
    train_data, test_data, train_label, test_label = train_test_split(train_corpus, class_label, test_size=0.1)



    del class_label,train_corpus
    gc.collect()

    print "开始模型训练:"
    predict_result = Bayesian_classifier(train_data, test_data, train_label)

    #AB,predict_result = train_AdaBoost_NB(train_data,test_data,train_label)
    #save_AdaBoostNB_model(AB)

    print "----------正在进行模型评估----------"
    evaluate(predict_result, test_label)
    del predict_result
    gc.collect()
    print "----------模型评估完成----------\n"

    time1 = time.time()
    t = time1 - time0
    print "总用时为： ",t
