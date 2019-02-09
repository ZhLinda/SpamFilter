#!/usr/bin/env python
# -*- coding:utf-8 -*-
from Preprocessing import *
from functions import *
from Feature_extraction import *
import os
from optparse import OptionParser

def Update_model_with_labeled_data(filename, Vector_type=3):

    time0 = time.time()

    cur_model_path = "../model_" + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    mkdir(cur_model_path)

    cur_derived_path = "../data/derived_" + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    mkdir(cur_derived_path)

    print "----------正在加载停用词表----------"
    stopWords = load_stop_words('../data/fixed/stopWord.txt')  # get the stopwords list
    print "----------停用词表加载完成----------\n"

    print "----------正在进行文本分词----------"
    get_data_segmented(stopWords, filename, "../data/derived")  # ../data/derived文件夹存储所有中间文件
    get_data_segmented(stopWords, filename, cur_derived_path)   # cur_derived_path文件夹存储本次更新的中间文件
    del stopWords
    gc.collect()
    print "------------文本分词完成------------\n"

    print "----------正在进行文本bigram切分----------"
    countNgram(2, filename, '../data/derived/train_ngram.txt')
    countNgram(2, filename, cur_derived_path + "/train_ngram.txt")
    print "------------文本bigram切分完成------------\n"

    print "----------正在合并分词结果----------"
    merge_words("../data/derived/train_seg.txt", "../data/derived/train_ngram.txt",
                "../data/derived/train_allwords.txt")
    merge_words(cur_derived_path + "/train_seg.txt", cur_derived_path + "/train_ngram.txt", cur_derived_path + "/train_allwords.txt")
    print "----------合并分词结果完成----------\n"

    print "----------正在加载类别信息----------"
    train_label = load_class_data(cur_derived_path + '/classLabel.txt')
    train_label_all = load_class_data("../data/derived/classLabel.txt")
    if os.path.isfile("../data/derived/classify_results.txt") == True:
        test_label_all = load_class_data("../data/derived/classify_results.txt")
    else:
        test_label_all = []
    print "----------类别信息加载完成----------\n"

    print "----------正在加载语料库----------"
    train_data = load_train_data(cur_derived_path + '/train_allwords.txt')
    print "----------语料库加载完成----------\n"


    Vocabulary = load_dictionary("../model/vocabulary.txt")
    dictionary = load_dictionary("../model/dictionary.txt")
    mat_dict_set = load_mat_dict("../model/mat_dict_set.txt")
    mat_dict_bag = load_mat_dict("../model/mat_dict_bag.txt")
    mat_dict_tfidf = load_mat_dict("../model/mat_dict_tfidf.txt")

    for i in range(len(train_label)):
        print i+1
        line = train_data[i].strip().split()
        label = train_label[i]
        hybrid = []
        for item in line:
            hybrid.append(item)
        if len(hybrid) == 0:
            hybrid.append("empty")

        '''根据文本更新Mat、模型和Vocabulary'''
        for word in hybrid:
            if word not in Vocabulary:
                Vocabulary.append(word)
                mat_dict_set.update({word: [0, 0]})
                mat_dict_bag.update({word: [0, 0]})
                mat_dict_tfidf.update({word: [0, 0]})

        save_dictionary(Vocabulary, cur_model_path + "/vocabulary.txt")  # 更新Vocabulary
        save_dictionary(Vocabulary, "../model/vocabulary.txt")

        new_mat_set = word2vec_setOfWords_single(hybrid, Vocabulary)
        new_mat_bag = word2vec_bagOfWords_single(hybrid, Vocabulary)
        new_mat_tfidf = word2vec_tfidf_single(hybrid, Vocabulary, len(train_label_all) + len(test_label_all))


        for i in range(len(Vocabulary)):
            term = Vocabulary[i]
            if new_mat_set[i] > 0:
                mat_dict_set[term][label] += new_mat_set[i]
            if new_mat_bag[i] > 0:
                mat_dict_bag[term][label] += new_mat_bag[i]
            if new_mat_tfidf[i] > 0:
                mat_dict_tfidf[term][label] += new_mat_tfidf[i]

        save_mat_dict(mat_dict_set, cur_model_path + "/mat_dict_set.txt")
        save_mat_dict(mat_dict_bag, cur_model_path + "/mat_dict_bag.txt")
        save_mat_dict(mat_dict_tfidf, cur_model_path + "/mat_dict_tfidf.txt")
        save_mat_dict(mat_dict_set, "../model/mat_dict_set.txt")
        save_mat_dict(mat_dict_bag, "../model/mat_dict_bag.txt")
        save_mat_dict(mat_dict_tfidf, "../model/mat_dict_tfidf.txt")

        temp = train_label_all #temp为到目前为止，所有的文本的label
        temp.extend(test_label_all)

        '''更新dictionary'''

        if (i + 1) % 1000 == 0:
            class_dict = get_class_dict(temp)
            class_df_list = stats_class_df(temp, class_dict)  # 得到正负样本数
            new_Vocabulary = feature_selection_ig(class_df_list, Vocabulary, mat_dict_bag)

            size = 3500
            dictionary = new_Vocabulary[:size]
            save_dictionary(dictionary, cur_model_path + "/dictionary.txt")
            save_dictionary(new_Vocabulary, cur_model_path + "/vocabulary.txt")
            save_dictionary(dictionary, "../model/dictionary.txt")
            save_dictionary(new_Vocabulary, "../model/vocabulary.txt")

            count_time(time0)

        '''根据新Mat和（可能的）新dictionary更新模型'''
        if Vector_type == 1:  # 词集方式
            Mat = mat_dict_set
        elif Vector_type == 2:  # 词袋方式
            Mat = mat_dict_bag
        elif Vector_type == 3:  # TFIDF方式
            Mat = mat_dict_tfidf

        count_time(time0)

        p1num = []
        p0num = []
        p0total = 2.0
        p1total = 2.0
        p_spam = sum(temp) / float(len(temp))
        p_ham = 1 - p_spam
        for term in dictionary:
            ham = Mat[term][0]
            spam = Mat[term][1]
            p1num.append(spam + 1)
            p0num.append(ham + 1)
            p1total += spam
            p0total += ham
        p_words_spam = np.array(p1num) / p1total
        p_words_ham = np.array(p0num) / p0total

        save_NB_model(p_words_ham, p_words_spam, p_spam, p_ham, cur_model_path)
        save_NB_model(p_words_ham, p_words_spam, p_spam, p_ham, "../model")


if __name__ == "__main__":
    usage = "usage:%prog [options] version=%prog 1.0"
    parser = OptionParser(usage=usage)
    parser.add_option("-f", "--filename", dest="filename")
    parser.add_option("-v", "--Vector_type", type="choice", choices=["1", "2", "3"], dest="Vector_type", default="3")
    options, args = parser.parse_args()

    filename = options.filename
    Vector_type = int(options.Vector_type)

    Update_model_with_labeled_data(filename, Vector_type)
