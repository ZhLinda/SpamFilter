#!/usr/bin/env Python
# coding=utf-8

from pyExcelerator import *
import xlrd


def write_Excel(ori_file, excel_file):
    fin = open(ori_file,"r")
    lines = [line.strip().decode('utf-8') for line in fin.readlines()]

    w = Workbook() #创建一个工作簿
    ws = w.add_sheet(u'sheet1')
    ws.write(0,0,u'服id')
    ws.write(0,1,u'pid')
    ws.write(0,2,u'uid')
    ws.write(0,3,u'timestamp')
    ws.write(0,4,u'message')
    ws.write(0,5,u'label')

    for i in range(len(lines)):
        line = lines[i].strip().split(" ",4)
        if len(line) < 5:
            line.append(" ")
        for j in range(5):
            ws.write(i+1,j,line[j])
    w.save(excel_file)

def read_Excel(excel_file,out_file):

    fout = open(out_file,"w")

    wb = xlrd.open_workbook(excel_file) #打开文件
    sheetNames = wb.sheet_names() #查看包含的工作表

    sh = wb.sheet_by_name(u'Sheet1')
    # 可以用cell_values(rowIndex, colIndex)对单元格进行操作

    for i in range(0,10000):
        rowValueList = sh.row_values(i)

        if rowValueList[5] == 1:
            label = 1
        else:
            label = 0
        fout.write(label.__str__() + " " + rowValueList[0].encode('utf-8') + " " + rowValueList[1].encode('utf-8') + " " + rowValueList[2].encode('utf-8') + " " + rowValueList[3].encode('utf-8') + " " + rowValueList[4].encode('utf-8') + "\n")



if __name__ == "__main__":
    #write_Excel("../data/to_be_labeled_4w.txt","../data/t.xls")
    read_Excel("../data/t.xlsx","../data/t.txt")
