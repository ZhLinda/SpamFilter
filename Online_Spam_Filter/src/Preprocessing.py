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

def get_data_segmented(stopWords,filename,prefix):
    fp = open(filename,'r')
    lines = fp.readlines()
    for line in lines:
        line = line.strip().split(' ',5) #以空格为分隔符，分隔4次

        writeStr(line[0], prefix + '/classLabel.txt')
        if len(line) < 6:
            message = " "
        else:
            message = line[5]

        uString = preProcessing(message) #uString是经过预处理的message的utf-8格式的字符串

        segWords = cutWords(uString,stopWords) #the segmented words cut by jieba

        writeListWords(segWords, prefix + '/train_seg.txt')
    del lines
    gc.collect()
    fp.close()


def countNgram(n,fp1,fp2):
    fin = open(fp1, 'r')
    fout = open(fp2, 'a+')
    lines = fin.readlines()
    #lines = [line.decode('utf-8') for line in fin.readlines()]
    stop_words = load_stop_words("../data/fixed/stopWord.txt")
    for oline in lines:
        line = oline.strip().split(' ',5) #以空格为分隔符，分隔4次
        if len(line) < 6:
            line.append(" ")
        uString = preProcessing(line[5]).strip().decode('utf-8').split()  # the uString is the preprocessed line in lines
        for word in uString:
            if word in stop_words:
                continue
            L = len(word)
            if L == 1:
                fout.write(word.encode('utf-8', 'ignore') + ' ')
                continue
            for i in range(0, L - n + 1):
                cur = word[i:i + n]
                flag = 0
                for ch in cur: #仅由数字或字母不能构成bigram
                    if is_Chinese(ch) == False:
                        flag = 0
                    else:
                        flag = 1
                        break
                if flag == 0:
                    continue
                else:
                    flag = 0
                fout.write(cur.encode('utf-8', 'ignore') + ' ')
                # cur = word[L-1:L]
                # fout.write(cur.encode('utf-8', 'ignore') + ' ')
        fout.write('\n')
    del lines
    del stop_words
    gc.collect()
    fin.close()
    fout.close()

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
    '''
    :param uStr: utf-8格式的字符串
    :return: 经过预处理的utf-8格式的字符串
    '''
    ustring = uStr.replace(' ','') # remove all blanks in the original text
    ustring = delete_facial_expr(ustring.decode('utf-8')) #删除所有表情符号
    ret = ""
    str = ustring.decode('utf-8')
    for i in str:
        ret += Q2B(i) #全角转半角
    ustring = ret.encode('utf-8')

    ustring = convert_num_char(ustring) #将数字的各种表示方式转换成正常数字
    #print ustring

    ret = string_to_list(ustring.decode('utf-8')) #删除了所有不是中文、英文、数字的字符
    msg = ''
    for key in ret:
        msg += key
    ustring = msg.encode('utf-8')
    return ustring
