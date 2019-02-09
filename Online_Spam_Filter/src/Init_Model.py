#!/usr/bin/env python
# -*- coding:utf-8 -*-
from Preprocessing import *
from functions import *
from Feature_extraction import *
from optparse import OptionParser


def Init_model(f_name, Vector_type=3):

    time0 = time.time()

    cur_model_path = "../model_" + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    mkdir(cur_model_path)

    print "----------正在加载停用词表----------"
    stopWords = load_stop_words('../data/fixed/stopWord.txt')  # get the stopwords list
    print "----------停用词表加载完成----------\n"

    print "----------正在进行文本分词----------"
    get_data_segmented(stopWords, f_name, "../data/derived")  # 使用JIEBA分词对所有文本进行分词
    del stopWords
    gc.collect()
    print "------------文本分词完成------------\n"

    print "----------正在进行文本bigram切分----------"
    countNgram(2, f_name, '../data/derived/train_ngram.txt')
    print "------------文本bigram切分完成------------\n"

    print "----------正在合并分词结果----------"
    merge_words("../data/derived/train_seg.txt", "../data/derived/train_ngram.txt",
                "../data/derived/train_allwords.txt")
    print "----------合并分词结果完成----------\n"

    print "----------正在加载类别信息----------"
    train_label = load_class_data('../data/derived/classLabel.txt')
    print "----------类别信息加载完成----------\n"

    print "----------正在加载语料库----------"
    train_data = load_train_data('../data/derived/train_allwords.txt')
    print "----------语料库加载完成----------\n"

    print "开始训练初始模型：\n"
    method = ['IG', 'CHI', 'MI']
    m = method[0]

    print "----------正在进行特征排序----------"

    class_dict = get_class_dict(train_label)
    print class_dict
    term_dict = get_term_dict(train_data)
    class_df_list = stats_class_df(train_label, class_dict)  # 得到正负样本数
    term_set = [term[0] for term in sorted(term_dict.items(), key=lambda x: x[1])]  # 按照每个term的df排序
    mat_set, mat_dict_set = term_class_mat_set(train_data, train_label, term_dict, class_dict)  # 原始词集中的词在不同类别下的文档频率
    mat_bag, mat_dict_bag = term_class_mat_bag(train_data, train_label, term_dict, class_dict)  # 原始词集中的词在不同类别下的出现总次数
    mat_tfidf, mat_dict_tfidf = term_class_mat_tfidf(train_data, train_label, term_dict,
                                                     class_dict)  # 原始词集中的词在不同类别下的TFIDF值

    save_mat_dict(mat_dict_set, "../model/mat_dict_set.txt")
    save_mat_dict(mat_dict_bag, "../model/mat_dict_bag.txt")
    save_mat_dict(mat_dict_tfidf, "../model/mat_dict_tfidf.txt")

    save_mat_dict(mat_dict_set, cur_model_path + "/mat_dict_set.txt")
    save_mat_dict(mat_dict_bag, cur_model_path + "/mat_dict_bag.txt")
    save_mat_dict(mat_dict_tfidf, cur_model_path + "/mat_dict_tfidf.txt")

    Vocabulary = feature_selection(m, class_df_list, mat_dict_set, mat_dict_bag,
                                   term_set)  # 按特征score进行了排序
    save_dictionary(Vocabulary, "../model/vocabulary.txt")
    save_dictionary(Vocabulary, cur_model_path + "/vocabulary.txt")
    print "Vocabulary的长度为：", len(Vocabulary)

    size = 3500
    dictionary = Vocabulary[:size]
    save_dictionary(dictionary, "../model/dictionary.txt")
    save_dictionary(dictionary, cur_model_path + "/vocabulary.txt")

    count_time(time0)

    '''根据Mat计算贝叶斯初始模型'''
    p1num = []
    p0num = []
    p0total = 2.0
    p1total = 2.0
    p_spam = sum(train_label) / float(len(train_label))
    p_ham = 1 - p_spam
    for term in dictionary:
        '''构造包含所有原始词汇信息的字典矩阵'''
        if Vector_type == 1:  # 词集方式
            Mat = mat_dict_set
        elif Vector_type == 2:  # 词袋方式
            Mat = mat_dict_bag
        elif Vector_type == 3:  # TFIDF方式
            Mat = mat_dict_tfidf
        ham = Mat[term][0]
        spam = Mat[term][1]
        p1num.append(spam + 1)
        p0num.append(ham + 1)
        p1total += spam
        p0total += ham

        del Mat
        gc.collect()
    p_words_spam = np.array(p1num) / p1total
    p_words_ham = np.array(p0num) / p0total
    print "初始模型构造完成"

    save_NB_model(p_words_ham, p_words_spam, p_spam, p_ham, "../model")
    save_NB_model(p_words_ham, p_words_spam, p_spam, p_ham, cur_model_path)

if __name__ == "__main__":

    usage = "usage:%prog [options] version=%prog 1.0"
    parser = OptionParser(usage=usage)
    parser.add_option("-f", "--filename", dest="filename")
    parser.add_option("-v", "--Vector_type", type="choice", choices=["1", "2", "3"], dest="Vector_type", default="3")
    options, args = parser.parse_args()

    filename = options.filename
    Vector_type = int(options.Vector_type)

    Init_model(filename, Vector_type)