#!/usr/bin/env python
# -*- coding:utf-8 -*-
import re
import time
import pickle
import os, subprocess, operator, argparse
from sys import argv
#import delims
import sys
from Chinese_handler import *
from Preprocessing import *
import gc

def countNgram(n):
    fin = open('../data/fixed/train_data_10w.txt', 'r')
    fout = open('../data/derived/train_ngram_10w.txt', 'a+')
    lines = fin.readlines()
    #lines = [line.decode('utf-8') for line in fin.readlines()]
    stop_words = load_stop_words()
    for oline in lines:
        line = oline.strip().split('\t',2)
        #print(line[0]) line[0] is the label!
        #fout.write(line[0].encode('utf-8') + ' ')
        uString = preProcessing(line[2]).decode('utf-8').split()  # the uString is the preprocessed line in lines
        for word in uString:
            if word in stop_words:
                continue
            L = len(word)
            if L == 1:
                fout.write(word.encode('utf-8', 'ignore') + ' ')
                continue
            for i in range(0, L - n + 1):
                cur = word[i:i + n]
                fout.write(cur.encode('utf-8', 'ignore') + ' ')
                # cur = word[L-1:L]
                # fout.write(cur.encode('utf-8', 'ignore') + ' ')
        fout.write('\n')
    del lines
    del stop_words
    gc.collect()
    fin.close()
    fout.close()

'''
下面对ngram分词后得到的grams进行筛选
'''

def load_dictionary():
    bigTable = {}
    fin = open('../data/derived/train_ngram_100w.txt', 'r')
    lines = [line.decode('utf-8') for line in fin.readlines()]
    for single_line in lines:
        words = single_line.split()
        length = len(words)
        print words[0] # the class label
        for i in range(1,length):
            if words[i] in bigTable:
                bigTable[words[i]] += 1
            else:
                bigTable.update({words[i]: 1})
    return bigTable


def neighborFilter():
    fin = open('../data/derived/train_ngram_100w.txt','r')
    fout = open('../data/derived/train_ngram_100w_new.txt','a+')
    lines = [line.decode('utf-8') for line in fin.readlines()]
    dict = load_dictionary()
    for line in lines:
        new_line = line
        words = line.split()
        length = len(words)
        for i in range(1,length-1):
            word1 = words[i]
            word2 = words[i+1]
            if word1 in dict and word2 in dict:
                freq1 = dict[word1]
                freq2 = dict[word2]
                if freq1 < 1.2 * freq2 and freq2 < 1.2 * freq1:
                    dict.update({word1: 0})
                    dict.update({word2: 0})
                    new_line = new_line.replace(word1+" ","").replace(word2+" ","")
                elif freq1 < 0.1 * freq2:
                    dict.update({word1: 0})
                    new_line = new_line.replace(word1 + " ","")
                elif freq2 < 0.1 * freq1:
                    dict.update({word2: 1})
                    new_line = new_line.replace(word2 + " ","")
        fout.write(new_line.encode('utf-8'))






'''
if __name__ == "__main__":
    countNgram(2)
    #load_dictionary()
    #neighborFilter()'''