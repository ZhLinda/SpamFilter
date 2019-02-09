#!/usr/bin/env python
# -*- coding:utf-8 -*-
from Preprocessing import *
from functions import *
from Feature_extraction import *
from optparse import OptionParser
import os

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


def E_step(p_words_spam, p_words_ham, p_spam, p_ham, hybrid,dictionary):

    print "E_step"
    voc = dict(zip(dictionary, range(len(dictionary))))
    mul_s = 1.0
    mul_h = 1.0
    for word in hybrid:
        if word in voc:
            mul_s = mul_s * p_words_spam[voc[word]]
            mul_h = mul_h * p_words_ham[voc[word]]

    up_spam = p_spam * mul_s
    up_ham = p_ham * mul_h
    down = up_spam + up_ham
    if down == 0:
        down = 1
    p_spam_d = up_spam / down
    p_ham_d = up_ham / down
    print p_spam_d, p_ham_d

    fp = open("../data/derived/classify_results_spam.txt", "a+")
    fp2 = open("../data/derived/classify_results_ham.txt", "a+")
    fp.write(p_spam_d.__str__() + "\n")
    fp2.write(p_ham_d.__str__() + "\n")
    fp.close()
    fp2.close()

    return p_spam_d, p_ham_d

def M_step(p_spam_d, p_ham_d, hybrid, Vector_type, update_model, model_choice):

    print "M_step"

    dictionary = load_dictionary(model_choice + "/dictionary.txt")
    mat_dict_set = load_mat_dict(model_choice + "/mat_dict_set.txt")
    mat_dict_bag = load_mat_dict(model_choice + "/mat_dict_bag.txt")
    mat_dict_tfidf = load_mat_dict(model_choice + "/mat_dict_tfidf.txt")

    train_label = load_class_data('../data/derived/classLabel.txt')
    pred_spam = load_predict_results("../data/derived/classify_results_spam.txt")
    Vocabulary = load_dictionary(model_choice + "/vocabulary.txt")


    p1num = []
    p0num = []
    p0total = 2.0
    p1total = 2.0

    p_spam = sum(train_label) + sum(pred_spam) / float(len(train_label) + len(pred_spam))
    p_ham = 1 - p_spam

    '''更新Mat、模型和Vocabulary'''
    for word in hybrid:
        if word not in Vocabulary:
            Vocabulary.append(word)
            mat_dict_set.update({word: [0,0]})
            mat_dict_bag.update({word: [0,0]})
            mat_dict_tfidf.update({word: [0,0]})

    new_mat_set = word2vec_setOfWords_single(hybrid, Vocabulary)
    new_mat_bag = word2vec_bagOfWords_single(hybrid, Vocabulary)
    new_mat_tfidf = word2vec_tfidf_single(hybrid, Vocabulary, len(train_label) + len(pred_spam))

    for i in range(len(Vocabulary)):
        term = Vocabulary[i]
        if new_mat_set[i] > 0:
            mat_dict_set[term][0] += new_mat_set[i] * p_ham_d
            mat_dict_set[term][1] += new_mat_set[i] * p_spam_d
        if new_mat_bag[i] > 0:
            mat_dict_bag[term][0] += new_mat_bag[i] * p_ham_d
            mat_dict_bag[term][1] += new_mat_bag[i] * p_spam_d
        if new_mat_tfidf[i] > 0:
            mat_dict_tfidf[term][0] += new_mat_tfidf[i] * p_ham_d
            mat_dict_tfidf[term][1] += new_mat_tfidf[i] * p_spam_d

    if update_model == 1: #若选择更新模型
        save_dictionary(Vocabulary, "../model/vocabulary.txt")  # 更新Vocabulary
        save_mat_dict(mat_dict_set, "../model/mat_dict_set.txt")
        save_mat_dict(mat_dict_bag, "../model/mat_dict_bag.txt")
        save_mat_dict(mat_dict_tfidf, "../model/mat_dict_tfidf.txt")

        save_dictionary(Vocabulary, model_choice + "/vocabulary.txt")
        save_mat_dict(mat_dict_set, model_choice + "/mat_dict_set.txt")
        save_mat_dict(mat_dict_bag, model_choice + "/mat_dict_bag.txt")
        save_mat_dict(mat_dict_tfidf, model_choice + "/mat_dict_tfidf.txt")

    if Vector_type == 1:  # 词集方式
        Mat = mat_dict_set
    elif Vector_type == 2:  # 词袋方式
        Mat = mat_dict_bag
    elif Vector_type == 3:  # TFIDF方式
        Mat = mat_dict_tfidf

    for term in dictionary:
        ham = Mat[term][0]
        spam = Mat[term][1]
        p1num.append(spam + 1)
        p0num.append(ham + 1)
        p1total += spam
        p0total += ham
    p_words_spam = np.array(p1num) / p1total
    p_words_ham = np.array(p0num) / p0total

    if update_model == 1:
        save_NB_model(p_words_ham, p_words_spam, p_spam, p_ham, model_choice)
        save_NB_model(p_words_ham, p_words_spam, p_spam, p_ham, "../model")


def Classify_new_data(new_data_filename, Vector_type, update_model=0, update_dictionary=0, model_choice="../model"):

    time0 = time.time()

    fnew = open(new_data_filename, "r")
    lines = fnew.readlines()

    for new_data in lines:
        p_words_ham, p_words_spam, p_spam, p_ham = load_NB_model(model_choice)
        dictionary = load_dictionary(model_choice + "/dictionary.txt")
        stopWords = load_stop_words('../data/fixed/stopWord.txt')

        line = new_data.strip().split(' ', 5)  # 以空格为分隔符，分隔4次

        writeStr(line[0], '../data/derived/test_label.txt')
        if len(line) < 6:
            message = "empty"
        else:
            message = line[5]

        ustring = preProcessing(message)
        segWords = cutWords(ustring, stopWords)  # the segmented words cut by jieba
        bigrams = get_bigram(stopWords, ustring)
        hybrid = []  # hybrid是由文本中所有分词构成的list
        for item in segWords:
            hybrid.append(item.strip())
        for item in bigrams:
            hybrid.append(item.strip().decode("utf-8"))
        if len(hybrid) == 0:
            hybrid.append("empty")
        writeListWords(hybrid, '../data/derived/hybrid.txt')

        '''使用EM算法计算每个文本所属类别的软概率'''

        p_spam_d, p_ham_d = E_step(p_words_spam, p_words_ham, p_spam, p_ham, hybrid, dictionary)

        M_step(p_spam_d, p_ham_d, hybrid, Vector_type, update_model, model_choice)

        count_time(time0)

        '''对新文本进行分类'''
        if p_spam_d > p_ham_d:
            label = 1
        else:
            label = 0

        writeStr(label.__str__(), '../data/derived/classify_results.txt')

        if update_model == 0:  # 不更新模型，只进行分类
            continue

        '''更新dictionary'''

        if update_dictionary == 1:
            Vocabulary = load_dictionary(model_choice + "/vocabulary.txt")  # 这里的Vocabulary已更新过
            mat_dict_bag = load_mat_dict(model_choice + "/mat_dict_bag.txt")  # 已更新
            train_label = load_class_data("../data/derived/classLabel.txt")
            test_label = load_class_data("../data/derived/classify_results.txt")
            train_label.extend(test_label)

            class_dict = get_class_dict(train_label)
            class_df_list = stats_class_df(train_label, class_dict)  # 得到正负样本数
            new_Vocabulary = feature_selection_ig(class_df_list, Vocabulary, mat_dict_bag)

            size = 3500
            dictionary = new_Vocabulary[:size]
            save_dictionary(dictionary, "../model/dictionary.txt")
            save_dictionary(new_Vocabulary, "../model/vocabulary.txt")
            save_dictionary(dictionary, model_choice + "/dictionary.txt")
            save_dictionary(new_Vocabulary, model_choice + "/vocabulary.txt")

            count_time(time0)



if __name__ == "__main__":
    usage = "usage:%prog [options] version=%prog 1.0"
    parser = OptionParser(usage=usage)
    parser.add_option("-f","--filename",dest="new_data_filename")
    parser.add_option("-v","--Vector_type",type="choice",choices=["1","2","3"],dest="Vector_type",default="3")
    parser.add_option("-m","--update_model", type="choice", choices=["0", "1"], dest="update_model", default="0")
    parser.add_option("-d","--update_dictionary",type="choice",choices=["0","1"],dest="update_dictionary",default="0")
    parser.add_option("-c","--model_choice",dest="model_choice",default="../model")
    options, args = parser.parse_args()

    filename = options.new_data_filename
    Vector_type = int(options.Vector_type)
    update_model = int(options.update_model)
    update_dictionary = int(options.update_dictionary)
    model_choice = options.model_choice

    Classify_new_data(filename, Vector_type, update_model, update_dictionary, model_choice)
