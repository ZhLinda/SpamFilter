#!/usr/bin/env python
# -*- coding:utf-8 -*-

from SVM_functions import *
from Feature_extraction import *
from Preprocessing import *
from Ngram import *
from sklearn import svm


import time




if __name__ == "__main__":

    time0 = time.time()

    print "----------正在加载停用词表----------"
    stopWords = load_stop_words('../data/fixed/stopWord.txt')  # get the stopwords list
    print "----------停用词表加载完成----------\n"

    print "----------正在进行文本分词----------"
    get_data_segmented(stopWords)  # 使用JIEBA分词对所有文本进行分词
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
    print len(feature_set)
    # size = 0.1 * len(feature_set)
    size = 3000
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

    #del train_corpus
    #gc.collect()

    fp3 = "../model/ext_feature_3.txt"
    length = len(tfidf_mat)
    for i in range(length):
        ans = sum(tfidf_mat[i])
        writeStr(ans.__str__(),fp3)

    del tfidf_mat
    gc.collect()
    print "----------新特征计算完成----------\n"


    print "----------正在进行word-to-vector转化----------"
    train_vectors = word2vec_tfidf(train_data, dictionary)

    print "----------word-to-vector转化完成----------\n"

    print "----------正在构建新的特征矩阵----------"


    l = len(train_data)
    del train_data
    gc.collect()

    fea_1 = load_extra_features("../model/ext_feature_1.txt",0,l)
    fea_2 = load_extra_features("../model/ext_feature_2.txt",0,l)
    fea_3 = load_extra_features("../model/ext_feature_3.txt",0,l)
    fea_4 = load_extra_features("../model/ext_feature_4.txt",0,l)
    fea_5 = load_extra_features("../model/ext_feature_5.txt",0,l)
    fea_6 = load_extra_features("../model/ext_feature_6.txt",0,l)

    #特征矩阵增加6列新特征
    train_vectors = np.column_stack((train_vectors,fea_1))
    train_vectors = np.column_stack((train_vectors,fea_2))
    train_vectors = np.column_stack((train_vectors,fea_3))
    train_vectors = np.column_stack((train_vectors,fea_4))
    train_vectors = np.column_stack((train_vectors,fea_5))
    train_vectors = np.column_stack((train_vectors,fea_6))

    del fea_1,fea_2,fea_3,fea_4,fea_6
    gc.collect()
    print "----------正在特征矩阵构建完成----------\n"

    print "----------正在进行特征矩阵规格化----------"

    min_max_scaler = preprocessing.MinMaxScaler() #将数据按列缩放到[0,1]之间
    train_vectors_scaled = min_max_scaler.fit_transform(train_vectors)

    save_train_vectors(train_vectors_scaled, "../model/train_vectors.txt")
    print "----------特征矩阵规格化完成并成功存储----------"


    del train_vectors
    gc.collect()

    time1 = time.time()
    t = time1 - time0
    print "总用时为： ",t


