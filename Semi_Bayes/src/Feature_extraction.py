#!/usr/bin/env python
# -*- coding:utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing
import math
from scipy import sparse
import time
import sys
import string
import numpy as np
import math
import gc

def merge_words():
    fin_1 = open("../data/derived/train_seg_10w.txt","r")
    fin_2 = open("../data/derived/train_ngram_10w.txt","r")
    fout = open("../data/derived/train_allwords_10w.txt","a+")
    lines_1 = [line.decode('utf-8') for line in fin_1.readlines()]
    lines_2 = [line.decode('utf-8') for line in fin_2.readlines()]
    for i in range(0, len(lines_1)):
        fout.write(lines_1[i].strip().encode('utf-8') + " " + lines_2[i].encode('utf-8'))
    del lines_1,lines_2
    gc.collect()
    fin_1.close()
    fin_2.close()


'''作废'''
def TF_IDF():
    #fin = open("../data/derived/train_allwords_1000.txt","r")
    corpus = load_train_data('../data/derived/train_seg_1000.txt')
    #lines = [line.decode('utf-8') for line in fin.readlines()]

    vectorizer = CountVectorizer() #将文本中的词语转换为词频矩阵
    X = vectorizer.fit_transform(corpus) #计算每个词语出现的次数
    word = vectorizer.get_feature_names() #获取词袋中所有文本关键词

    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    weight = tfidf.toarray()

    #sklearn中已经进行了正则化，因此这里白写了...
    '''formalization = []
    for i in range(len(weight)):
        tmp = sum(weight[i])
        tmp = math.sqrt(tmp)
        #formalization.append(tmp)
        for j in range(len(weight[0])):
            weight[i][j] = weight[i][j] / tmp'''

    '''
    for i in range(len(weight)):
        print u"-------这里输出第",i,u"篇文本的词语tf-idf权重------"
        for j in range(len(word)):
            print word[j],weight[i][j]
            #fout.write(word[j].encode('utf-8') + " " + weight[i][j].tostring().encode('utf-8'))
    '''
    return weight


def load_class_data(filename):
    class_list = []
    for line in open(filename,'r').readlines():
        class_list.append(int(line.strip().decode('utf-8')))
    return class_list

def load_train_data(filename):
    data_list = []
    for line in open(filename,'r').readlines():
        data_list.append(line.strip().decode('utf-8'))
    return data_list

def get_class_dict(doc_class_list):
    class_set = sorted(list(set(doc_class_list)))
    class_dict = dict(zip(class_set,range(len(class_set))))
    return class_dict

def get_term_dict(doc_terms_list):
    term_set_dict = {}
    for doc_terms in doc_terms_list:
        for term in doc_terms.split():
            if term in term_set_dict:
                term_set_dict[term] += 1
            else:
                term_set_dict.update({term: 1})

    term_set_list = sorted(term_set_dict.keys())
    term_set_dict = dict(zip(term_set_list, range(len(term_set_list))))
    return term_set_dict

def stats_class_df(doc_class_list, class_dict): #正负样本数
    class_df_list = [0] * len(class_dict)
    for doc_class in doc_class_list:
        class_df_list[class_dict[doc_class]] += 1
    return class_df_list



def stats_term_df(doc_terms_list,term_dict):
    term_df_dict = {}.fromkeys(term_dict.keys(), 0)
    for term in term_dict:
        for doc_terms in doc_terms_list:
            if term in doc_terms:
                term_df_dict[term] += 1
    return term_df_dict

def stats_term_class_df(doc_term_list, doc_class_list, term_dict, class_dict):
    term_class_df_mat = np.zeros((len(term_dict), len(class_dict)), np.float64)
    for k in range(len(doc_class_list)):
        class_index = class_dict[doc_class_list[k]]
        doc_terms = doc_term_list[k]
        for term in doc_terms.split():
            term_index = term_dict[term]
            term_class_df_mat[term_index][class_index] += 1
    return term_class_df_mat

def feature_selection_ig(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat #每个词在两个类别中出现的次数
    B = np.array([(sum(x) - x).tolist() for x in A]) #每个词在相反类别中出现的次数
    C = np.tile(class_df_list, (A.shape[0], 1)) - A #每个词在两个类别中不出现的次数
    '''
    numpy.tile([0,0],(1,3))#在列方向上重复[0,0]3次，行1次  
    '''
    N = sum(class_df_list) #文档总数
    D = N - A - B - C #每个词在相反类别中不出现的次数
    term_df_array = np.sum(A, axis=1)
    class_set_size = len(class_df_list)

    p_t = term_df_array / N
    #print p_t
    p_not_t = 1 - p_t
    p_c_t_mat = (A + 1) / (A + B + class_set_size)
    p_c_not_t_mat = (C + 1) / (C + D + class_set_size)
    p_c_t = np.sum(p_c_t_mat * np.log(p_c_t_mat), axis=1)
    p_c_not_t = np.sum(p_c_not_t_mat * np.log(p_c_not_t_mat), axis=1)
    term_score_array = p_t * p_c_t + p_not_t * p_c_not_t
    #print "The term_score_array is :"
    #print term_score_array
    sorted_term_score_index = term_score_array.argsort()[:: -1] #argsort()返回的是数组值从小到大的索引，但[::-1]则会将索引逆序排列，即从大到小排列
    #print "The sorted_term_score_index is :"
    #print sorted_term_score_index
    term_set_fs = [term_set[index] for index in sorted_term_score_index]
    del A,B,C,D,term_df_array,p_t,p_not_t,p_c_t_mat,p_c_not_t_mat,p_c_t,p_c_not_t,term_score_array,sorted_term_score_index
    gc.collect()
    return term_set_fs


def feature_selection_chi2(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    D = N - A - B - C

    up = (A*D - B*C) * (A*D - B*C)
    down = (A + B) * (C + D)
    term_score_matrix = (up + 1) / (down + len(class_df_list))
    term_score_array = []
    for row in term_score_matrix:
        term_score_array.append(row[0])
    #print "The term_score_array is :"
    #print term_score_array
    sorted_term_score_index = np.array(term_score_array).argsort()[:: -1]
    #print "The sorted_term_score_index is :"
    #print sorted_term_score_index
    term_set_fs = [term_set[index] for index in sorted_term_score_index]
    del A,B,C,D,up,down,term_score_array,term_score_matrix,sorted_term_score_index
    gc.collect()
    return term_set_fs

def feature_selection_mi(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list,(A.shape[0],1)) - A
    N = sum(class_df_list)
    class_set_size = 2

    term_score_matrix = np.log(((A+1.0)*N) / ((A+C) * (A+B+class_set_size)))
    term_score_max_array = np.array([max(x) for x in term_score_matrix])
    sorted_term_score_index = term_score_max_array.argsort()[:: -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index]
    del A,B,C,term_score_matrix,term_score_max_array,sorted_term_score_index
    gc.collect()
    return term_set_fs


def word2vec_setOfWords(words,dictionary): # 词集模型，出现为1,不出现为0
    voc = dict(zip(dictionary,dictionary))
    doc_dict_mat = np.zeros((len(words),len(dictionary)), np.float64)
    for i in range(len(words)):
        for word in words[i].split():
            if word in voc:
                doc_dict_mat[i][dictionary.index(word)] = 1



    del voc
    gc.collect()
    return doc_dict_mat


def word2vec_bagOfWords(words,dictionary): # 词袋模型
    voc = dict(zip(dictionary,dictionary)) #增加访问速度
    doc_dict_mat = np.zeros((len(words), len(dictionary)), np.float64)
    for i in range(len(words)):
        for word in words[i].split():# words为分词的全部结果， words[i]为一条文本的分词结果，word为单个词
            if word in voc:
                doc_dict_mat[i][dictionary.index(word)] += 1

    '''权重向量的规格化'''
    for i in range(len(words)):  # 对每一篇文档
        regu = 1
        for j in range(len(dictionary)):  # 对该文档中的每一个特征词
            regu += doc_dict_mat[i][j] * doc_dict_mat[i][j]
        regu = math.sqrt(regu)
        doc_dict_mat[i] = doc_dict_mat[i] / regu

    del voc
    gc.collect()
    return doc_dict_mat

def word2vec_tfidf(words,dictionary): # 以TFIDF表示特征权重
    voc = dict(zip(dictionary,dictionary))
    doc_dict_mat = np.zeros((len(words),len(dictionary)), np.float64)
    term_df_dict = stats_term_df(words,voc) #原文本中所有词的文档频率的字典
    lenD = len(words)
    for i in range(len(words)):
        for word in words[i].split():
            if word in voc:
                tf = words[i].split().count(word)
                w = np.log(tf+1) * np.log((lenD+1) / term_df_dict[word])
                doc_dict_mat[i][dictionary.index(word)] = w


    del voc,term_df_dict
    gc.collect()
    return doc_dict_mat


def feature_selection(doc_term_list, doc_class_list, fs_method):
    class_dict = get_class_dict(doc_class_list)
    term_dict = get_term_dict(doc_term_list)
    class_df_list = stats_class_df(doc_class_list,class_dict) #得到正负样本数
    term_class_df_mat = stats_term_class_df(doc_term_list, doc_class_list, term_dict, class_dict) #字典对应的词在不同类别下的词频
    term_set = [term[0] for term in sorted(term_dict.items(), key = lambda x:x[1])] #按照每个term的df排序
    print "The size of initial feature set is :", len(term_set)
    if fs_method == 'IG':
        print 'IG'
        term_set_fs = feature_selection_ig(class_df_list, term_set, term_class_df_mat)
        fout = open("../data/derived/IG_features_10w.txt","w")
        for term in term_set_fs:
            fout.write(term.encode('utf-8') + "\n")

    elif fs_method == 'CHI':
        print 'CHI'
        term_set_fs = feature_selection_chi2(class_df_list, term_set, term_class_df_mat)
        fout = open("../data/derived/CHI2_features_10w.txt", "w")
        for term in term_set_fs:
            fout.write(term.encode('utf-8') + "\n")

    elif fs_method == 'MI':
        print 'MI'
        term_set_fs = feature_selection_mi(class_df_list, term_set, term_class_df_mat)
        fout = open("../data/derived/MI_features_10w.txt","w")
        for term in term_set_fs:
            fout.write(term.encode('utf-8') + "\n")

    del class_dict,term_dict,class_df_list,term_class_df_mat,term_set
    gc.collect()
    return term_set_fs


def validation_classifier(traindata,testdata,trainlabel,testlabel):
    #method = ['CHI','IG']
    method = ['IG','CHI','MI']
    for m in method:
        t1 = time.time()
        feature_set = feature_selection(traindata,trainlabel,m)
        print len(feature_set)
        dictionary = feature_set[:500000]
        #for term in dictionary:
            #print term

        '''fea_train = word2vec(train_data,dictionary)
        fea_test = word2vec(test_data,dictionary)
        print fea_train.shape
        print fea_test.shape'''




'''
if __name__ == "__main__":
    merge_words()
    class_label = load_class_data('../data/derived/classLabel_100w.txt')
    train_corpus = load_train_data('../data/derived/train_seg_100w.txt')

    #交叉验证，验证集占总训练集的20%
    train_data, test_data, train_label, test_label = train_test_split(train_corpus, class_label,test_size=0.1)
    validation_classifier(train_data, test_data, train_label, test_label)

'''