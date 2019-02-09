#!/usr/bin/env python
# -*- coding:utf-8 -*-
import gc

def get_10000_data_for_check():
    fp = open("../data/derived/labeled_data.txt", "r")
    fp2 = open("../data/labeled_data_check.txt", "w")
    lines = fp.readlines()
    for i in range(10000):
        fp2.write(lines[i])


def count_category_number():
    fp = open("../model/results.txt")
    count_0 = 0
    count_1 = 0
    for i in fp.readlines():
        if i.strip() == "0":
            count_0 += 1
        else:
            count_1 += 1
    print "The number of hams is: ",count_0
    print "The number of spams is: ",count_1

def set_categorys_apart():
    fin = open("../data/derived/labeled_data.txt", "r")
    fout1 = open("../data/labeled_data_check_spam.txt","w")
    fout2 = open("../data/labeled_data_check_ham.txt","w")
    lines = fin.readlines()
    for line in lines:
        ustring = line.strip().split(" ",5)
        if len(ustring) < 6:
            ustring.append(" ")
            print ustring[2]
        label = ustring[0]
        if label == "1":
            fout1.write(ustring[0].strip() + " " + ustring[5].strip() + "\n")
        else:
            fout2.write(ustring[0].strip() + " " + ustring[5].strip() + "\n")

def merge_by_type():

    for i in range(1,31):

        if i >= 1 and i <= 9:
            fin_1 = open("../data/fixed/pirate9000000" + i.__str__() + "_barrage_record_new", "r")
            fin_2 = open("../data/fixed/pirate9000000" + i.__str__() + "_chat", "r")
            fin_3 = open("../data/fixed/pirate9000000" + i.__str__() + "_chat_guild", "r")
            fin_4 = open("../data/fixed/pirate9000000" + i.__str__() + "_chat_world", "r")

            fout = open("../data/fixed/pirate9000000" + i.__str__() + "_total", "w")

        else:
            fin_1 = open("../data/fixed/pirate900000" + i.__str__() + "_barrage_record_new", "r")
            fin_2 = open("../data/fixed/pirate900000" + i.__str__() + "_chat", "r")
            fin_3 = open("../data/fixed/pirate900000" + i.__str__() + "_chat_guild", "r")
            fin_4 = open("../data/fixed/pirate900000" + i.__str__() + "_chat_world", "r")

            fout = open("../data/fixed/pirate900000" + i.__str__() + "_total", "w")


        fout.writelines(fin_1.readlines())
        fout.writelines(fin_2.readlines())
        fout.writelines(fin_3.readlines())
        fout.writelines(fin_4.readlines())
        fin_1.close()
        fin_2.close()
        fin_3.close()
        fin_4.close()
        fout.close()


def merge_by_time():
    fin_1 = open("../data/man_labeled/ham.txt", "r")
    fin_2 = open("../data/man_labeled/spam.txt", "r")
    #fin_3 = open("../data/fixed/pirate90000023_chat_guild", "r")
    #fin_4 = open("../data/fixed/pirate90000023_chat_world", "r")
    fout = open("../data/all.txt", "w")

    lines_tuples = []

    lines_1 = fin_1.readlines()
    for line in lines_1:
        line = line.strip().split(' ', 5)  # 以空格为分隔符，分隔4次
        if len(line) < 6:
            line.append(" ")
        lines_tuples.append((line[0],line[1],line[2],line[3],line[4],line[5]))
    fin_1.close()
    del lines_1
    gc.collect()

    lines_2 = fin_2.readlines()
    for line in lines_2:
        line = line.strip().split(' ', 5)  # 以空格为分隔符，分隔4次
        if len(line) < 6:
            line.append(" ")
        lines_tuples.append((line[0], line[1], line[2], line[3], line[4],line[5]))
    fin_2.close()
    del lines_2
    gc.collect()

    '''
    lines_3 = fin_3.readlines()
    for line in lines_3:
        line = line.strip().split(' ', 4)  # 以空格为分隔符，分隔4次
        if len(line) < 5:
            line.append(" ")
        lines_tuples.append((line[0], line[1], line[2], line[3], line[4]))
    fin_3.close()
    del lines_3
    gc.collect()

    lines_4 = fin_4.readlines()
    for line in lines_4:
        line = line.strip().split(' ', 4)  # 以空格为分隔符，分隔4次
        if len(line) < 5:
            line.append(" ")
        lines_tuples.append((line[0], line[1], line[2], line[3], line[4]))
    fin_4.close()
    del lines_4
    gc.collect()
    '''

    lines_tuples = sorted(lines_tuples, key=lambda line: line[4])

    for item in lines_tuples:
        fout.write(item[0] + " " + item[1] + " " + item[2] + " " + item[3] + " " + item[4] + " " + item[5] + "\n")
    fout.close()

def plus_zero_label():
    fin = open("../data/ham_.txt","r")
    fout = open("../data/ham.txt","w")
    lines = fin.readlines()
    for line in lines:
        fout.write("0 " + line)



if __name__ == "__main__":
    #count_category_number()
    #set_categorys_apart()
    merge_by_time()
    #plus_zero_label()