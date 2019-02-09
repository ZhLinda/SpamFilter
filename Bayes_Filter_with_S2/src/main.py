#!/usr/bin/env python
# -*- coding:utf-8 -*-

from NaiveBayes import *
import time
from Handle_S2 import *

if __name__ == "__main__":

    time0 = time.time()

    print "----------正在加载停用词表----------"
    stopWords = load_stop_words('../data/fixed/stopWord.txt')  # get the stopwords list
    print "----------停用词表加载完成----------\n"

    print "----------正在进行文本分词----------"
    get_data_segmented(stopWords, "../data/fixed/train_data_1w.txt")  # 使用JIEBA分词对所有文本进行分词
    del stopWords
    gc.collect()
    print "------------文本分词完成------------\n"

    print "----------正在进行文本bigram切分----------"
    countNgram(2, '../data/fixed/train_data_1w.txt', '../data/derived/train_ngram.txt')
    print "------------文本bigram切分完成------------\n"

    print "----------正在合并分词结果----------"
    merge_words("../data/derived/train_seg.txt", "../data/derived/train_ngram.txt","../data/derived/train_allwords.txt")
    print "----------合并分词结果完成----------\n"

    print "----------正在加载类别信息----------"
    # class_label = load_class_data('../data/derived/classLabel_5000.txt')
    class_label = load_class_data('../data/derived/classLabel.txt')
    print "----------类别信息加载完成----------\n"

    print "----------正在加载语料库----------"
    # train_corpus = load_train_data('../data/derived/train_allwords_5000.txt')
    train_corpus = load_train_data('../data/derived/train_allwords.txt')
    print "----------语料库加载完成----------\n"



    # 交叉验证，验证集占总训练集的10%
    train_data, test_data, train_label, test_label = train_test_split(train_corpus, class_label, test_size=0.1)



    del class_label
    gc.collect()

    print "开始模型训练:"
    method = ['IG', 'CHI', 'MI']
    m = method[0]

    flag = 1
    if flag == 1:  # 重新选择特征，训练模型
        print "----------正在进行特征选择----------"
        feature_set = feature_selection(train_data, train_label, m)
        print len(feature_set)
        # size = 0.1 * len(feature_set)
        size = 3000
        dictionary = feature_set[:int(size)]  # 选取前1000个词词作为特征
        del feature_set
        gc.collect()
        # for term in dictionary:
        # print term

        save_dictionary(dictionary, "../model/dictionary.txt")

        print "----------特征选择完成----------\n"

        print "----------正在计算新特征的值----------"
        tfidf_mat = word2vec_tfidf(train_corpus, dictionary)  # 所有文本的，而不仅仅是训练集的

        # del train_corpus
        # gc.collect()

        fp3 = "../model/ext_feature_3.txt"
        length = len(tfidf_mat)
        for i in range(length):
            ans = sum(tfidf_mat[i])
            writeStr(ans.__str__(), fp3)
        del tfidf_mat
        gc.collect()

        print "----------新特征计算完成----------\n"

        print "----------正在进行word-to-vector转化----------"
        train_vectors_S1 = word2vec_tfidf(train_data, dictionary)


        save_train_vectors(train_vectors_S1, "../model/train_vectors_S1.txt")
        print "----------word-to-vector转化完成----------\n"






        print "----------正在进行模型训练----------"


        p_words_ham, p_words_spam, p_spam = trainNB(train_vectors_S1, train_label)


        save_NB_model(p_words_ham, p_words_spam, p_spam, dictionary)

        print "----------模型训练完成----------\n"
        #del train_vectors_scaled_S1

    else:  # 载入储存的模型
        p_words_ham, p_words_spam, p_spam, dictionary = load_NB_model()#
        train_vectors_scaled_S1 = load_train_vectors(len(train_data),len(dictionary),".../model/train_vectors_S1.txt")

    print "----------正在进行testing----------"

    test_vector = word2vec_tfidf(test_data, dictionary)

    del dictionary
    gc.collect()

    '''
    test_vector_scaled = min_max_scaler.transform(test_vector)
    del test_vector
    gc.collect()
    '''
    s2_theta, s2_sigma, scaler = Handle_S2(train_data, train_label)

    l = len(test_data)
    test_vectors_S2 = np.zeros((l, 5), np.float64)

    fea_1 = load_extra_features("../model/ext_feature_1.txt", 0, l)
    #fea_2 = load_extra_features("../model/ext_feature_2.txt", 0, l)
    fea_3 = load_extra_features("../model/ext_feature_3.txt", 0, l)
    #fea_4 = load_extra_features("../model/ext_feature_4.txt", 0, l)
    #fea_5 = load_extra_features("../model/ext_feature_5.txt", 0, l)
    fea_6 = load_extra_features("../model/ext_feature_6.txt", 0, l)
    fea_7 = load_extra_features("../model/ext_feature_7.txt", 0, l)
    fea_8 = load_extra_features("../model/ext_feature_8.txt", 0, l)

    for i in range(l):
        test_vectors_S2[i][0] = fea_1[i]
        #test_vectors_S2[i][1] = fea_2[i]
        test_vectors_S2[i][1] = fea_3[i]
        #test_vectors_S2[i][3] = fea_4[i]
        #test_vectors_S2[i][4] = fea_5[i]
        test_vectors_S2[i][2] = fea_6[i]
        test_vectors_S2[i][3] = fea_7[i]
        test_vectors_S2[i][4] = fea_8[i]

    #del fea_1, fea_2, fea_3, fea_4, fea_5, fea_6, fea_7, fea_8
    gc.collect()

    test_vectors_S2_scaled = scaler.transform(test_vectors_S2)

    classify_result = []
    for i in range(len(test_vector)):
        vec2classify = test_vector[i] #s1 part
        vec_s2 = test_vectors_S2[i] #s2 part
        label, t1, t2 = classify(vec_s2, vec2classify, p_words_ham, p_words_spam, p_spam, s2_theta, s2_sigma)
        classify_result.append(label)
    print "----------testing完成----------\n"

    del test_vector
    gc.collect()


    print "----------正在进行模型评估----------"
    evaluate(classify_result, test_label)
    del classify_result
    gc.collect()
    print "----------模型评估完成----------\n"

    time1 = time.time()
    t = time1 - time0
    print "总用时为： ",t
