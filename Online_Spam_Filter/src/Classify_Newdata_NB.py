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


def Classify_new_data(new_data_filename, Vector_type=3, update_model=0, update_dictionary=0, model_choice="../model"):
    time0 = time.time()

    fnew = open(new_data_filename, "r")
    lines = fnew.readlines()
    for new_data in lines:

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

        if update_model == 0:  # 不更新模型，只进行分类
            continue

        '''根据新文本更新Mat、模型和Vocabulary'''
        for word in hybrid:
            if word not in Vocabulary:
                Vocabulary.append(word)
                mat_dict_set.update({word: [0, 0]})
                mat_dict_bag.update({word: [0, 0]})
                mat_dict_tfidf.update({word: [0, 0]})

        save_dictionary(Vocabulary, model_choice + "/vocabulary.txt")  # 更新Vocabulary
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

