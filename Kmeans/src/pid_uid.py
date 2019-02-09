#!/usr/bin/env python
# -*- coding:utf-8 -*-

if __name__ == "__main__":
    for i in range(10,49):
        f1 = open("../data/fixed/pid_uid_pirate900000" + i.__str__(), "r")
        f2 = open("../data/fixed/pirate900000" + i.__str__() + "_barrage_record","r")
        f3 = open("../data/fixed/pirate900000" + i.__str__() + "_barrage_record_new","w")
        pid_uid_pairs = f1.readlines()
        dic = {}
        for item in pid_uid_pairs:
            pair = item.strip().split("\t")
            #print len(pair)
            dic[pair[1]]=pair[0]
        #print dic
        lines = f2.readlines()
        for line in lines:
            parts = line.strip().split(" ",4)
            if len(parts) < 5:
                parts.append(" ")
            uid = parts[2]
            if(uid in dic):
                pid = dic[uid]
            else:
                pid = "0"
            parts[1] = pid
            f3.write(parts[0] + " " + parts[1] + " " + parts[2] + " " + parts[3] + " " + parts[4] + "\n")