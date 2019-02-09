#!/usr/bin/env python
# -*- coding:utf-8 -*-
print (__doc__)

from SVM_functions import *
from Feature_extraction import *
from Preprocessing import *
from Ngram import *
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

from sklearn.datasets import load_boston
from sklearn.linear_model import (LinearRegression, RandomizedLogisticRegression, Ridge,
                                  Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time

ranks = {}


def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order * np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)  # 将ranks中的每一个元素都保留两位小数
    return dict(zip(names, ranks))


if __name__ == "__main__":

    time0 = time.time()

    print "----------正在加载停用词表----------"
    stopWords = load_stop_words('../data/fixed/stopWord.txt')  # get the stopwords list
    print "----------停用词表加载完成----------\n"

    print "----------正在进行文本分词----------"
    get_data_segmented(stopWords,"../data/fixed/train_data_5000.txt")  # 使用JIEBA分词对所有文本进行分词
    del stopWords
    gc.collect()
    print "------------文本分词完成------------\n"

    print "----------正在进行文本bigram切分----------"
    countNgram(2,'../data/fixed/train_data_5000.txt','../data/derived/train_ngram.txt')
    print "------------文本bigram切分完成------------\n"

    print "----------正在合并分词结果----------"
    merge_words("../data/derived/train_seg.txt","../data/derived/train_ngram.txt","../data/derived/train_allwords.txt")
    print "----------合并分词结果完成----------\n"


    print "----------正在加载类别信息----------"
    train_label = load_class_data('../data/derived/classLabel.txt')
    print "----------类别信息加载完成----------\n"

    print "----------正在加载语料库----------"
    train_data = load_train_data('../data/derived/train_allwords.txt')
    print "----------语料库加载完成----------\n"



    method = ['IG', 'CHI', 'MI']
    m = method[0]

    L = len(train_data)


    print "----------正在进行特征选择----------"
    feature_set = feature_selection(train_data, train_label, m)
    del train_label
    gc.collect()
    print len(feature_set)
    # size = 0.1 * len(feature_set)
    size = 4000
    dictionary = feature_set[:int(size)]  # 选取前3000个词词作为特征
    del feature_set
    gc.collect()

    save_dictionary(dictionary,"../model/dictionary.txt")
    # for term in dictionary:
    # print term
    print "----------特征选择完成----------\n"

    print "----------正在计算新特征的值----------"
    #tfidf_mat = word2vec_tfidf(train_corpus, dictionary) #所有文本的，而不仅仅是训练集的
    tfidf_mat = word2vec_tfidf(train_data,dictionary)
    del dictionary
    gc.collect()
    #del train_corpus
    #gc.collect()

    fp3 = "../model/ext_feature_3.txt"
    length = len(tfidf_mat)
    for i in range(length):
        ans = sum(tfidf_mat[i])
        writeStr(ans.__str__(),fp3)


    print "----------新特征计算完成----------\n"


    dictionary = load_dictionary("../model/new_dict.txt")
    print "----------正在进行word-to-vector转化----------"
    #train_label = load_class_data('../data/derived/classLabel.txt')
    #train_data = load_train_data('../data/derived/train_allwords.txt')
    train_vectors = word2vec_tfidf(train_data,dictionary)
    #train_vectors = tfidf_mat

    print "----------word-to-vector转化完成----------\n"

    print "----------正在构建新的特征矩阵----------"

    l = len(train_data)
    #del tfidf_mat
    gc.collect()
    del train_data
    gc.collect()

    #fea_1 = load_extra_features("../model/ext_feature_1.txt", 0, l)
    #fea_2 = load_extra_features("../model/ext_feature_2.txt", 0, l)
    fea_3 = load_extra_features("../model/ext_feature_3.txt", 0, l)
    #fea_4 = load_extra_features("../model/ext_feature_4.txt", 0, l)
    #fea_5 = load_extra_features("../model/ext_feature_5.txt", 0, l)
    fea_6 = load_extra_features("../model/ext_feature_6.txt", 0, l)
    #fea_7 = load_extra_features("../model/ext_feature_7.txt", 0, l)
    #fea_8 = load_extra_features("../model/ext_feature_8.txt", 0, l)

    # 特征矩阵增加6列新特征
    #train_vectors = np.column_stack((train_vectors, fea_1))
    #train_vectors = np.column_stack((train_vectors, fea_2))
    train_vectors = np.column_stack((train_vectors, fea_3))
    #train_vectors = np.column_stack((train_vectors, fea_4))
    #train_vectors = np.column_stack((train_vectors, fea_5))
    train_vectors = np.column_stack((train_vectors, fea_6))
    #train_vectors = np.column_stack((train_vectors, fea_7))
    #train_vectors = np.column_stack((train_vectors, fea_8))

    #del fea_1, fea_2, fea_3, fea_4, fea_5, fea_6, fea_7, fea_8
    gc.collect()
    print "----------正在特征矩阵构建完成----------\n"

    print "----------正在进行特征矩阵规格化----------"

    min_max_scaler = preprocessing.MinMaxScaler()  # 将数据按列缩放到[0,1]之间
    train_vectors_scaled = min_max_scaler.fit_transform(train_vectors)

    save_train_vectors(train_vectors_scaled, "../model/train_vectors.txt")
    print "----------特征矩阵规格化完成并成功存储----------"



    '''
    randomized_logistic = RandomizedLogisticRegression()
    randomized_logistic.fit(train_vectors_scaled,train_label)

    print "Features sorted by their rank:"
    result = sorted(zip(map(lambda x: round(x, 4), randomized_logistic.scores_), names))
    print result
    fp = open("result.txt", "w")
    for item in result:
        fp.write(item[0].__str__() + " " + item[1].encode("utf-8") + "\n")
    '''
    del train_vectors,train_vectors_scaled
    gc.collect()

    time1 = time.time()
    t = time1 - time0
    print "总用时为： ", t


