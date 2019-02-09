#!/usr/bin/env python
# -*- coding:utf-8 -*-
from numpy import *
import jieba
import jieba.analyse
import gc
from Chinese_handler import *

def load_stop_words():
    stop = [line.strip().decode('utf-8') for line in open('../data/fixed/stopWord.txt').readlines()]
    return stop

def get_data_segmented(stopWords):
    fp = open("../data/fixed/train_data_10w.txt",'r')
    lines = fp.readlines()
    for line in lines:
        #the following "line"s are just a line of the text
        line = line.strip().split('\t')
        #print(line[1])
        #open("../data/derived/train_seg_1000.txt",'a+').write(line[0].encode('utf-8') + ' ')
        writeStr(line[1],'../data/derived/classLabel_10w.txt')
        #if line[0] == '1':
            #writeStr(line[1],'../data/derived/rubbishMsg_1000.txt')
        uString = preProcessing(line[2]) #the uString is the preprocessed line in lines
        segWords = cutWords(uString,stopWords) #the segmented words cut by jieba
        writeListWords(segWords,'../data/derived/train_seg_10w.txt')
    del lines
    gc.collect()
    fp.close()



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
    ret = string_to_list(ustring.decode('utf-8')) #删除了所有不是中文、英文、数字的字符
    msg = ''
    for key in ret:
        msg += key
    ustring = msg.encode('utf-8')
    ustring = ustring.replace('x元','x价钱')
    ustring = ustring.replace('x日','x日期')
    ustring = ustring.replace('www','网站')
    return ustring

#def load_test_data(stopWords):

'''
if __name__ == "__main__":
    stopWords = load_stop_words() #get the stopwords list
    get_data_segmented(stopWords)
    #load_test_data(stopWords)'''
