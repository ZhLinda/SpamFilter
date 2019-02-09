#!/usr/bin/env python
# -*- coding:utf-8 -*-
from numpy import *
import jieba
import jieba.analyse
import gc
from Chinese_handler import *
import re
import sys

def load_stop_words(filename):
    stop = [line.strip().decode('utf-8') for line in open(filename).readlines()]
    return stop

def get_data_segmented(stopWords):
    fp = open("../data/fixed/train_data_5000.txt",'r')
    lines = fp.readlines()
    for line in lines:
        #the following "line"s are just a line of the text
        line = line.strip().split('\t',2)
        #print(line[1])
        #open("../data/derived/train_seg_1000.txt",'a+').write(line[0].encode('utf-8') + ' ')
        writeStr(line[1],'../data/derived/classLabel_5000.txt')
        #if line[0] == '1':
            #writeStr(line[1],'../data/derived/rubbishMsg_1000.txt')
        uString, cnt_filtered = preProcessing(line[2]) #the uString is the preprocessed line in lines

        total_ord = total_char_ord(line[2].replace(' ','').decode('utf-8'))

        get_extra_features(uString,cnt_filtered,total_ord) #计算除词向量外的其他特征

        segWords = cutWords(uString,stopWords) #the segmented words cut by jieba

        writeListWords(segWords,'../data/derived/train_seg_5000.txt')
    del lines
    gc.collect()
    fp.close()

def total_char_ord(str):
    '''
    字符 unicode 数值
    '''
    total = 0.0
    for c in str:
        total += ord(c)
    if len(str) == 0:
        return 0
    else:
        return total


if __name__ == "__main__":
    str = "【只需ち0圆，赠VIP橙装武将；646OOO176】适裏是估佬*榊秘旳榊域适裏姷"
    str2 = "郊区吊打一区 那一区算什么 难民区吗"
    print str.replace(' ','').decode('utf-8')
    ans = total_char_ord(str.replace(' ','').decode('utf-8'))
    print str2.replace(' ','').decode('utf-8')
    ans2 = total_char_ord(str2.replace(' ','').decode('utf-8'))
    print ans,ans2


def get_extra_features(ustring,cnt_filtered,total_ord):
    fp1 = "../model/ext_feature_1.txt"
    fp2 = "../model/ext_feature_2.txt"
    #fp3 = "../model/ext_feature_3.txt"
    fp4 = "../model/ext_feature_4.txt"
    fp5 = "../model/ext_feature_5.txt"
    fp6 = "../model/ext_feature_6.txt"
    fp7 = "../model/ext_feature_7.txt"
    fea_1 = len(re.findall('[a-zA-Z0-9][a-zA-Z0-9]+',ustring)) #计算每个样本中长度大于等于2的字母数字串的个数
    fea_2 = len(re.findall('[a-zA-Z0-9][a-zA-Z0-9][a-zA-Z0-9]+',ustring)) #计算每个样本中长度大于等于3的字母数字串的个数
    fea_4 = len(ustring.decode('utf-8')) #计算样本长度 len("fxx很棒帮哦".decode("utf-8")) == 7
    fea_5 = total_ord
    fea_6 = cnt_filtered
    st = re.findall("[a-zA-Z0-9]+",ustring)
    st.sort(key = lambda x: len(x))
    st = list(reversed(st))
    if st == []:
        fea_7 = 0
    else:
        fea_7 = len(st[0])
    writeStr(fea_1.__str__(),fp1)
    writeStr(fea_2.__str__(),fp2)
    writeStr(fea_4.__str__(),fp4)
    writeStr(fea_5.__str__(),fp5)
    writeStr(fea_6.__str__(),fp6)
    writeStr(fea_7.__str__(),fp7)

    return fp1,fp2,fp4


def cutWords(str,stopWords):
    seg_list = jieba.cut(str,cut_all = False)
    seg_words = []
    for i in seg_list:
        if i not in stopWords:
            seg_words.append(i)
    return seg_words


def writeStr(str,filename):
    fout = open('../data/' + filename, 'a+')
    fout.write(str+'\n')
    fout.close()

def writeListWords(seg_list,filename): #put the segmented words of a separate line into a file
    fout = open('../data/' + filename, 'a+')
    word_list = list(seg_list)
    out_str = ''
    for word in word_list:
        out_str += word
        out_str += ' '
    fout.write(out_str.encode('utf-8') + '\n')
    fout.close()

def preProcessing(uStr):
    ustring = uStr.replace(' ','') # remove all blanks in the original text
    temp = ""
    for ch in ustring:
        temp += Q2B(ch) #全角转半角
    ustring = temp
    ret = string_to_list(ustring.decode('utf-8')) #删除了所有不是中文、英文、数字的字符
    msg = ''
    for key in ret:
        msg += key
    ustring = msg.encode('utf-8')
    ustring = ustring.replace('x元','x价钱')
    ustring = ustring.replace('x日','x日期')
    ustring = ustring.replace('www','网站')
    return ustring


