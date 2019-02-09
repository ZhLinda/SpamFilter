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

def average_char_ord(str):
    '''
    平均字符 unicode 数值
    '''
    total = 0.0
    for c in str:
        total += ord(c)
    if len(str) == 0:
        return 0
    else:
        return total / len(str)





def get_extra_features(message,ustring,cnt_filtered,total_ord,ave_ord):
    '''
    :param message: 未经预处理的原文本 utf-8格式
    :param ustring: 经过预处理的文本 utf-8格式
    :param cnt_filtered: 经过预处理，被删除的字节长度
    :param total_ord:
    :param ave_ord:
    :return:
    '''
    fp1 = "../model/ext_feature_1.txt"
    fp2 = "../model/ext_feature_2.txt"
    #fp3 = "../model/ext_feature_3.txt"
    fp4 = "../model/ext_feature_4.txt"
    fp5 = "../model/ext_feature_5.txt"
    fp6 = "../model/ext_feature_6.txt"
    fp7 = "../model/ext_feature_7.txt"
    fp8 = "../model/ext_feature_8.txt"
    fea_1 = len(re.findall('[a-zA-Z0-9][a-zA-Z0-9]+',ustring)) #计算每个样本中长度大于等于2的字母数字串的个数
    fea_2 = len(re.findall('[a-zA-Z0-9][a-zA-Z0-9][a-zA-Z0-9]+',ustring)) #计算每个样本中长度大于等于3的字母数字串的个数
    fea_4 = len(message.decode('utf-8')) #计算未经预处理的样本长度 len("fxx很棒帮哦".decode("utf-8")) == 7
    fea_5 = total_ord #文本的编码总大小
    fea_6 = cnt_filtered #过滤掉的字符数
    st = re.findall("[a-zA-Z0-9]+",ustring)
    st.sort(key = lambda x: len(x))
    st = list(reversed(st))
    if st == []:
        fea_7 = 0
    else:
        fea_7 = len(st[0]) #最长的字母、数字串个数
    fea_8 = ave_ord #文本的平均编码大小
    writeStr(fea_1.__str__(),fp1)
    writeStr(fea_2.__str__(),fp2)
    writeStr(fea_4.__str__(),fp4)
    writeStr(fea_5.__str__(),fp5)
    writeStr(fea_6.__str__(),fp6)
    writeStr(fea_7.__str__(),fp7)
    writeStr(fea_8.__str__(),fp8)

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

def count_filtered(message): #计算未经预处理的文本中有多少非中文、英文、数字的特殊符号
    ustring = string_to_list(message.decode('utf-8'))
    msg = ''
    for key in ustring:
        msg += key
    cnt_filtered = len(message) - len(msg.encode('utf-8'))
    return cnt_filtered

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

if __name__ == "__main__":
    str = "d%h>b50亓=50万亓保V12-1忆垠+绅蒋进阶识,906-585-699開ㄝ台"
    ustring = preProcessing(str)
    cnt = count_filtered(str)
    print str
    print ustring
    print cnt
