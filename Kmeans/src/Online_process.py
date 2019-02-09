#!/usr/bin/env python
# -*- coding:utf-8 -*-
from Preprocessing import *
import time
from Ngram import *
from Feature_extraction import *
from functions import *


def Init_model(f_name, Vector_type):

    cur_model_path = "../model_" + time.strftime('%Y%m%d', time.localtime(time.time()))
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



    '''构造包含所有原始词汇信息的字典矩阵'''
    if Vector_type == 1:  # 词集方式
        Mat = mat_dict_set
    elif Vector_type == 2:  # 词袋方式
        Mat = mat_dict_bag
    elif Vector_type == 3:  # TFIDF方式
        Mat = mat_dict_tfidf

    count_time(time0)

    '''根据Mat计算贝叶斯初始模型'''
    p1num = []
    p0num = []
    p0total = 2.0
    p1total = 2.0
    p_spam = sum(train_label) / float(len(train_label))
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

    save_NB_model(p_words_ham, p_words_spam, p_spam, p_ham, "../model")
    save_NB_model(p_words_ham, p_words_spam, p_spam, p_ham, cur_model_path)





def Update_model_with_labeled_data(filename, Vector_type):

    cur_model_path = "../model_" + time.strftime('%Y%m%d', time.localtime(time.time()))
    mkdir(cur_model_path)

    cur_derived_path = "../data/derived_" + time.strftime('%Y%m%d', time.localtime(time.time()))
    mkdir(cur_derived_path)

    print "----------正在加载停用词表----------"
    stopWords = load_stop_words('../data/fixed/stopWord.txt')  # get the stopwords list
    print "----------停用词表加载完成----------\n"

    print "----------正在进行文本分词----------"
    get_data_segmented(stopWords, filename, "../data/derived")  # 使用JIEBA分词对所有文本进行分词
    get_data_segmented(stopWords, filename, cur_derived_path)
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
    test_label_all = load_class_data("../data/derived/classify_results.txt")
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




def get_bigram(stopWords, ustring):
    bigrams = []
    uString = ustring.strip().decode('utf-8').split()  # the uString is the preprocessed line in lines
    for word in uString:
        if word in stopWords:
            continue
        L = len(word)
        if L == 1:
            bigrams.append(word.encode('utf-8'))
            continue
        for i in range(0, L - 2 + 1):
            cur = word[i:i + 2]
            flag = 0
            for ch in cur:  # 仅由数字或字母不能构成bigram
                if is_Chinese(ch) == False:
                    flag = 0
                else:
                    flag = 1
                    break
            if flag == 0:
                continue
            else:
                flag = 0
            bigrams.append(cur.encode('utf-8'))
    return bigrams


def Classify_new_data(new_data, Vector_type, update_model=0, update_dictionary=0, model_choice="../model"):

    p_words_ham, p_words_spam, p_spam, p_ham = load_NB_model(model_choice)
    Vocabulary = load_dictionary(model_choice + "/vocabulary.txt")
    dictionary = load_dictionary(model_choice + "/dictionary.txt")
    stopWords = load_stop_words('../data/fixed/stopWord.txt')
    mat_dict_set = load_mat_dict(model_choice + "/mat_dict_set.txt")
    mat_dict_bag = load_mat_dict(model_choice + "/mat_dict_bag.txt")
    mat_dict_tfidf = load_mat_dict(model_choice + "/mat_dict_tfidf.txt")

    line = new_data.strip().split(' ', 5)  # 以空格为分隔符，分隔4次

    writeStr(line[0], '../data/derived/test_label.txt')
    if len(line) < 6:
        message = "empty"
    else:
        message = line[5]

    ustring = preProcessing(message)
    segWords = cutWords(ustring, stopWords)  # the segmented words cut by jieba
    bigrams = get_bigram(stopWords, ustring)
    hybrid = []
    for item in segWords:
        hybrid.append(item.strip())
    for item in bigrams:
        hybrid.append(item.strip().decode("utf-8"))
    if len(hybrid) == 0:
        hybrid.append("empty")
    writeListWords(hybrid, '../data/derived/hybrid.txt')


    train_label = load_class_data('../data/derived/classLabel.txt')
    test_label = load_class_data("../data/derived/test_label.txt")

    '''对新文本进行分类'''
    # print hybrid
    vec_set = word2vec_setOfWords_single(hybrid, dictionary)
    save_new_vector(vec_set, "../data/derived/vec_set.txt")
    vec_bag = word2vec_bagOfWords_single(hybrid, dictionary)
    save_new_vector(vec_bag, "../data/derived/vec_bag.txt")
    vec_tfidf = word2vec_tfidf_single(hybrid, dictionary, len(train_label) + len(test_label))
    save_new_vector(vec_tfidf, "../data/derived/vec_tfidf.txt")

    if Vector_type == 1:
        test_vector = vec_set
    elif Vector_type == 2:
        test_vector = vec_bag
    elif Vector_type == 3:
        test_vector = vec_tfidf
    # print test_vector

    label, t1, t2 = classify(test_vector, p_words_ham, p_words_spam, p_spam)
    writeStr(label.__str__(), '../data/derived/classify_results.txt')

    if update_model == 0: #不更新模型，只进行分类
        return

    '''根据新文本更新Mat、模型和Vocabulary'''
    for word in hybrid:
        if word not in Vocabulary:
            Vocabulary.append(word)
            mat_dict_set.update({word: [0, 0]})
            mat_dict_bag.update({word: [0, 0]})
            mat_dict_tfidf.update({word: [0, 0]})

    save_dictionary(Vocabulary, model_choice + "/vocabulary.txt") #更新Vocabulary
    save_dictionary(Vocabulary, "../model/vocabulary.txt")



    new_mat_set = word2vec_setOfWords_single(hybrid, Vocabulary)
    new_mat_bag = word2vec_bagOfWords_single(hybrid, Vocabulary)
    new_mat_tfidf = word2vec_tfidf_single(hybrid, Vocabulary, len(train_label) + len(test_label))

    del train_label, test_label
    gc.collect()

    for i in range(len(Vocabulary)):
        term = Vocabulary[i]
        if new_mat_set[i] > 0:
            mat_dict_set[term][label] += new_mat_set[i]
        if new_mat_bag[i] > 0:
            mat_dict_bag[term][label] += new_mat_bag[i]
        if new_mat_tfidf[i] > 0:
            mat_dict_tfidf[term][label] += new_mat_tfidf[i]

    save_mat_dict(mat_dict_set, model_choice + "/mat_dict_set.txt")
    save_mat_dict(mat_dict_bag, model_choice + "/mat_dict_bag.txt")
    save_mat_dict(mat_dict_tfidf, model_choice + "/mat_dict_tfidf.txt")
    save_mat_dict(mat_dict_set, "../model/mat_dict_set.txt")
    save_mat_dict(mat_dict_bag, "../model/mat_dict_bag.txt")
    save_mat_dict(mat_dict_tfidf, "../model/mat_dict_tfidf.txt")

    train_label = load_class_data("../data/derived/classLabel.txt")
    test_label = load_class_data("../data/derived/classify_results.txt")
    train_label.extend(test_label)

    '''更新dictionary'''

    if update_dictionary == 1:

        class_dict = get_class_dict(train_label)
        class_df_list = stats_class_df(train_label, class_dict)  # 得到正负样本数
        new_Vocabulary = feature_selection_ig(class_df_list, Vocabulary, mat_dict_bag)

        size = 3500
        dictionary = new_Vocabulary[:size]
        save_dictionary(dictionary, model_choice + "/dictionary.txt")
        save_dictionary(new_Vocabulary, model_choice + "/vocabulary.txt")
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
    p_spam = sum(train_label) / float(len(train_label))
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

    save_NB_model(p_words_ham, p_words_spam, p_spam, p_ham, model_choice)
    save_NB_model(p_words_ham, p_words_spam, p_spam, p_ham, "../model")


if __name__ == "__main__":
    time0 = time.time()

    Vector_type = 2
    '''
    1: 词集方式； 2：词袋方式； 3：TFIDF方式
    '''


    f_init = "../data/man_labeled/initial_labeled_data.txt"
    #Init_model(f_init, Vector_type)

    Update_model_with_labeled_data("../data/new_data/test.txt")


    '''标记新数据，同时更新模型'''
    f_new_data = "../data/new_data/test.txt"
    test_corpus = open(f_new_data, "r").readlines()
    print len(test_corpus)
    for count in range(len(test_corpus)):
        Classify_new_data(test_corpus[count], Vector_type, update_model=1, update_dictionary=1)

    test_label = load_class_data("../data/derived/test_label.txt")
    classify_results = load_class_data("../data/derived/classify_results.txt")
    evaluate(classify_results, test_label)

