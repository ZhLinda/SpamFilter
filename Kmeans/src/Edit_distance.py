#!/usr/bin/env python
# -*- coding:utf-8 -*-
import gc
from Feature_extraction import *
import Levenshtein
import string

def edit_distance():
    fin = open("../data/derived/server_23_by_time.txt", "r")
    fout = open("../data/derived/edit_distance_results", "w")

    distance_list = []
    last_message_dict = {} #格式为{pid:[mes1,mes2,...,mes50]} 消息池容量为50

    lines = fin.readlines()
    for line in lines:
        line = line.strip().split(' ', 4)  # 以空格为分隔符，分隔4次
        if len(line) < 5:
            line.append(" ")

        server_id = line[0]
        pid = line[1]
        uid = line[2]
        timestamp = line[3]
        #print timestamp
        message = line[4]

        if len(message) < 16:
            distance_list.append(0)
            continue

        if "收人" in message or "招人" in message:
            distance_list.append(0)
            continue

        if pid not in last_message_dict: #该条消息的发送者之前尚未发送信息
            distance_list.append(0)
            last_message_dict.update({pid:[message,]}) #此时消息池中仅有新添加的一条消息

        else: #该条消息的发送者之前发送过消息，且该消息储存在last_sentence_dict中

            #判断当前消息和消息池中的消息的距离小于10的个数
            last_sentence_list = last_message_dict[pid]
            count = 0
            for sentence in last_sentence_list:
                if Levenshtein.distance(sentence, message) < 10:
                    count += 1
            if count >= 3:
                distance_list.append(1)
            else:
                distance_list.append(0)

            #更新消息池
            size = len(last_message_dict[pid])
            if size < 50:
                last_message_dict[pid].append(message)
            else:
                del last_message_dict[pid][0]
                last_message_dict[pid].append(message)


    print len(distance_list)
    for item in distance_list:
        fout.write(item.__str__() + "\n")



    fin.close()
    fout.close()

    merge_words( "../data/derived/edit_distance_results", "../data/derived/server_23_by_time.txt","../data/derived/labeled_server_23.txt")

    del lines
    gc.collect()

if __name__ == "__main__":
    edit_distance()
    merge_words("../data/derived/all_results.txt","../data/derived/labeled_server_23.txt", "../data/derived/need_to_compare.txt")