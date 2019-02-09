#!/usr/bin/env python
# -*- coding:utf-8 -*-
from numpy import *
from functions import *
from Feature_extraction import *
from Preprocessing import *
from Ngram import *
import time

if __name__ == "__main__":

    for i in range(1,31):
        print i

        if i >= 1 and i <= 9:
            fin = open("../data/fixed/pirate9000000" + i.__str__() + "_total", "r")
            fout_spam = open("../data/derived/pirate9000000" + i.__str__() + "_spam", "w")
            fout_unknown = open("../data/derived/pirate9000000" + i.__str__() + "_unknown", "w")

        elif i >= 10 and i <= 30:
            fin = open("../data/fixed/pirate900000" + i.__str__() + "_total", "r")
            fout_spam = open("../data/derived/pirate900000" + i.__str__() + "_spam", "w")
            fout_unknown = open("../data/derived/pirate900000" + i.__str__() + "_unknown", "w")


        #fin = open("../data/randomly_choose_data_10000.txt","r")
        #fout_spam = open("../data/spam.txt","w")
        #fout_unknown = open("../data/ham.txt","w")

        spamwords_list_1 = ["ぇ", "ち", "ぢ", "ろ", "ⅥP", "開ㄝ台"]
        spamwords_list_2 = ["亓", "代充", "代冲", "代宠", "到付", "礼包", "大礼", "礼苞", "禮苞", "福利", "福禄", "购", "元宝", "员宝",
                            "元寶", "圆寳", "园饱", "橙装", "橙裝", "橙色装备", "神装", "武将", "神将", "橙将", "橙將", "进阶石", "vip",
                            "VIP", "Vip", "银币", "银", "先货", "先貨", "宪貨", "折扣", "充值", "冲直", "优惠", "沖值", "折kou",
                            "茺直", "铳直", "澫", "萭", "蜇筘", "哲扣", "折口", "开局", "開局", "壕礼", "簧色", "茺值", "省钱",
                            "扣裙", "洮宝", "橙卡", "美欐旳", "榊秘旳", "衧啯", "适裏", "免费",
                            "哲抠", "员堡"]

        lines = fin.readlines()
        fin.close()

        for i in range(len(lines)):
            line = lines[i].strip().split(' ', 4)  # 以空格为分隔符，分隔4次

            server_id = line[0]
            pid = line[1]
            uid = line[2]
            timestamp = line[3]
            if len(line) < 5:
                message = " "
            else:
                message = line[4]

            label = 0

            for word in spamwords_list_1:
                if word in message:
                    label = 1
                    break

            if label == 1:
                fout_spam.write(label.__str__() + " " + lines[i].strip() + "\n")
                continue

            ustring = preProcessing(message)
            st = re.findall("[a-zA-Z0-9]+", ustring)
            st.sort(key=lambda x: len(x))
            st = list(reversed(st))
            if st == []:
                maxlen = 0
            else:
                maxlen = len(st[0])  # 最长的字母、数字串个数
            if maxlen >= 6:
                for word in spamwords_list_2:
                    if word in ustring:
                        label = 1
                        break
            #print i


            if label == 1:
                fout_spam.write(label.__str__() + " " + lines[i].strip() + "\n")
            else:
                fout_unknown.write(lines[i].strip() + "\n")

        fout_unknown.close()
        fout_spam.close()