# -*- coding: utf-8 -*-
import csv
import numpy as np
import util

filename = './data/data_six.csv' #使用的原始训练数据

def read_result():
    csvfile=open('result.csv','r')
    reader=csv.reader(csvfile)
    b_result=[]
    for item in reader:
        b_result.append(float(item[1]))
    list_r=util.from_666_to_140(filename,b_result)
    util.to_csv('single',list_r)
    
def main():
    read_result()
    
    
if __name__=='__main__':
    main()