# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:13:04 2018

@author: charlie
"""

import os
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

def make_wavenet_batch(filename):
    data = genfromtxt(filename, delimiter=',', skip_header=1)
    data = data[:,1:]
    data = data.T
    temp_mean = np.mean(data,axis=0)
    temp_std = np.std(data,axis=0)
    data = (data - temp_mean)/temp_std
    input_data = data[:-1,:]
    input_label = data[1:,:]
    input_data.shape = (1,) + input_data.shape
    input_label.shape = (1,) + input_label.shape
    input_data = tf.convert_to_tensor(input_data,dtype=tf.float32)
    input_label = tf.convert_to_tensor(input_label,dtype=tf.float32)
    return input_data, input_label,temp_mean,temp_std

def make_lstm_batch(filename,duration,shuffle=False):
    data = genfromtxt(filename, delimiter=',', skip_header=1)
    data = data[:,1:]
    data = data.T
    temp_mean = np.mean(data,axis=0)
    temp_std = np.std(data,axis=0)
    data = (data - temp_mean)/temp_std
    data_len = data.shape[0]
    channels = data.shape[1]
    data = tf.convert_to_tensor(data,name='raw_data',dtype=tf.float32)
    epoch_size = data_len - duration
    i = tf.train.range_input_producer(epoch_size, shuffle=shuffle).dequeue()
    x = tf.slice(data, [i,0], [duration, channels])
    y = tf.slice(data, [i+1,0], [duration, channels])
    x = tf.expand_dims(x,0)
    y = tf.expand_dims(y,0)
    return x,y,temp_mean,temp_std

def make_lstm4_batch(filename,duration,shuffle=False):
    #读取csv文件
    data = genfromtxt(filename, delimiter=',', skip_header=1)
    #第一列为日期，不要
    data = data[:,1:]
#    to_tile = data[:,-22:-2]
#    tile = np.tile(to_tile,[1,3])
#    data = np.hstack((data,tile))
    #转置
    data = data.T
    target = data[-1,:]
    data[data==0] = 1
    
    #让序列中的后一个值减去前一个值
    data = -1*(data[1:,:] / data[:-1,:])
    
    data = 2/(1+np.exp(data))-1
    
#    data[data>0] = np.log(data[data>0])
#    data[data<0] = -np.log(-data[data<0])
    #得到数据维度
    data_len = data.shape[0]
    channels = data.shape[1]
    #变成常数tensor
    data = tf.convert_to_tensor(data,name='raw_data',dtype=tf.float32)
    #使用tensorflow自带的自动编号器来制作输入队列
    epoch_size = data_len - duration
    i = tf.train.range_input_producer(epoch_size, shuffle=shuffle).dequeue()
    x = tf.slice(data, [i,0], [duration, channels])
    y = tf.slice(data, [i+1,0], [duration, channels])
    x = tf.expand_dims(x,0)
    y = tf.expand_dims(y,0)
    return x,y,target
    
def make_lstm5_batch(filename,duration,shuffle=False):
    #读取csv文件
    data = genfromtxt(filename, delimiter=',', skip_header=1)
    #第一列为日期，不要
    data = data[:,1:]
    #转置
    data = data.T
    data[data==0] = 1
    #让序列中的后一个值减去前一个值
    data = data[1:,:] / data[:-1,:]
    
    data = np.log(data)
    #得到数据维度
    data_len = data.shape[0]
    channels = data.shape[1]
    #变成常数tensor
    data = tf.convert_to_tensor(data,name='raw_data',dtype=tf.float32)
    #使用tensorflow自带的自动编号器来制作输入队列
    epoch_size = data_len - duration # 69 - 20 = 49
    i = tf.train.range_input_producer(epoch_size, shuffle=shuffle).dequeue()
    x = tf.slice(data, [i,0], [duration, channels])
    y = tf.slice(data, [i+1,0], [duration, channels])
    x = tf.expand_dims(x,0)
    y = tf.expand_dims(y,0)
    return x,y

def make_lstm5_batch2(filename,duration,shuffle=False):
    #读取csv文件
    data = genfromtxt(filename, delimiter=',', skip_header=1)
    #第一列为日期，不要 -201710 data
    data = data[:,1:-1]
    
    #转置
    data = data.T
    data[data==0] = 1
    #让序列中的后一个值减去前一个值    
    data = -1*(data[1:,:] / data[:-1,:])
    
    data = 1/(1+np.exp(data))
    #得到数据维度
    data_len = data.shape[0]
    channels = data.shape[1]
    #变成常数tensor
    data = tf.convert_to_tensor(data,name='raw_data',dtype=tf.float32)
    #使用tensorflow自带的自动编号器来制作输入队列
    epoch_size = data_len - duration # 69 - 20 = 49
    i = tf.train.range_input_producer(epoch_size, shuffle=shuffle).dequeue()
    x = tf.slice(data, [i,0], [duration, channels])
    y = tf.slice(data, [i+1,0], [duration, channels])
    x = tf.expand_dims(x,0)
    y = tf.expand_dims(y,0)
    return x,y

def make_lstm5_batch2_1(filename,duration,shuffle=False):
    #读取csv文件
    data = genfromtxt(filename, delimiter=',', skip_header=1)
    #第一列为日期，不要 -201710 data
    data = data[:,1:]
    
    #转置
    data = data.T
    data[data==0] = 1
    #让序列中的后一个值减去前一个值    
    data = -1*(data[1:,:] / data[:-1,:])
    
    data = 2/(1+np.exp(data))-1
    #得到数据维度
    data_len = data.shape[0]
    channels = data.shape[1]
    #变成常数tensor
    data = tf.convert_to_tensor(data,name='raw_data',dtype=tf.float32)
    #使用tensorflow自带的自动编号器来制作输入队列
    epoch_size = data_len - duration # 69 - 20 = 49
    i = tf.train.range_input_producer(epoch_size, shuffle=shuffle).dequeue()
    x = tf.slice(data, [i,0], [duration, channels])
    y = tf.slice(data, [i+1,0], [duration, channels])
    x = tf.expand_dims(x,0)
    y = tf.expand_dims(y,0)
    return x,y

def make_lstm5_batch3(filename,duration,batch,shuffle=False):
    #读取csv文件
    data = genfromtxt(filename, delimiter=',', skip_header=1)
    #第一列为日期，不要 -201710 data
    data = data[:,1:-1]
    #转置
    data = data.T
    data[data==0] = 1
    #让序列中的后一个值减去前一个值    
    data = -1*(data[1:,:] / data[:-1,:])
    
    data = 1/(1+np.exp(data))
        #得到数据维度
    data_len = data.shape[0]
#    channels = data.shape[1]
    channels = 1

    #变成常数tensor
    data = tf.convert_to_tensor(data,name='raw_data',dtype=tf.float32)
    #使用tensorflow自带的自动编号器来制作输入队列
    epoch_size = data_len - duration # 69 - 20 = 49
    i = tf.train.range_input_producer(epoch_size, shuffle=shuffle).dequeue()
    x = tf.slice(data, [i,batch], [duration, channels])
    y = tf.slice(data, [i+1,batch], [duration, channels])
    x = tf.expand_dims(x,0)
    y = tf.expand_dims(y,0)
#    print x,y
    return x,y

def make_lstm3_batch(filename,duration,shuffle=False):
    data = genfromtxt(filename, delimiter=',', skip_header=1)
    data = data[:,1:]
    data = data.T
    date_mean = np.mean(data,axis=0)
    date_std = np.std(data,axis=0)
    data_sum = np.sum(data,axis=1)
    data_sum[:5] = 206
    data_sum = data_sum[1:]/data_sum[:-1]-1
    data_sum = np.insert(data_sum,0,1)
    data = (data - date_mean)/date_std
    data_len = data.shape[0]
    channels = data.shape[1]
    data = tf.convert_to_tensor(data,name='raw_data',dtype=tf.float32)
    data_sum = tf.convert_to_tensor(data_sum,name='raw_sum',dtype=tf.float32)
    data_sum = tf.expand_dims(data_sum,1)
    data = tf.concat(1,[data,data_sum])
    epoch_size = data_len - duration
    i = tf.train.range_input_producer(epoch_size, shuffle=shuffle).dequeue()
    x = tf.slice(data, [i,0], [duration, channels+1])
    y = tf.slice(data, [i+1,0], [duration, channels+1])
    x = tf.expand_dims(x,0)
    y = tf.expand_dims(y,0)
    return x,y,date_mean,date_std
    
def get_test_data3(filename,duration):
    data = genfromtxt(filename, delimiter=',', skip_header=1)
    data = data[:,1:]
    data = data.T
    temp_mean = np.mean(data,axis=0)
    temp_std = np.std(data,axis=0)
    data_sum = np.sum(data,axis=1)
    data_sum[:5] = 206
    data_sum = data_sum[1:]/data_sum[:-1]-1
    data_sum = np.insert(data_sum,0,1)
    data_sum.shape = data_sum.shape + (1,)
    data = (data - temp_mean)/temp_std
    data = np.concatenate((data,data_sum),axis=1)
    output = data[-duration:,:]
    output.shape = (1,duration,-1)
    return output  
    

def get_lstm_batch(filename,duration,batch_size):
    data = genfromtxt(filename, delimiter=',', skip_header=1)
    data = data[:,1:]
    data = data.T
#    temp_sum = np.sum(data,axis=0)
#    data = data / temp_sum
    temp_mean = np.mean(data,axis=0)
    temp_std = np.std(data,axis=0)
    data = (data - temp_mean)/temp_std
    prediction = data[1:,:]
    data = data[:-1,:]
    ed = data.shape[0] - duration
    choice = np.random.randint(0,ed,batch_size)
    output_batch = np.zeros((batch_size,duration,data.shape[1]))
    label_batch = np.zeros_like(output_batch)
    for i in range(batch_size):
        index = choice[i]
        output_batch[i,:,:] = data[index:index+duration,:]
        label_batch[i,:,:] = prediction[index:index+duration,:]
    return output_batch,label_batch
    
def get_test_data(filename,duration):
    data = genfromtxt(filename, delimiter=',', skip_header=1)
    data = data[:,1:]
    data = data.T
    temp_mean = np.mean(data,axis=0)
    temp_std = np.std(data,axis=0)
    data = (data - temp_mean)/temp_std
    output = data[-duration:,:]
    output.shape = (1,duration,-1)
    return output
    
#得到测试数据    
def get_test2_data(filename,duration):
    data = genfromtxt(filename, delimiter=',', skip_header=1)
    data = data[:,1:]
    data = data.T
    ori_data = data.copy()
    data = data[1:,:] - data[:-1,:]
    data[data>0] = np.log(data[data>0])
    data[data<0] = -np.log(-data[data<0])
    output = data[-duration:,:]
    output.shape = (1,duration,-1)
    return output, ori_data[-duration:,:]

def get_test3_data(filename,duration):
    data = genfromtxt(filename, delimiter=',', skip_header=1)
    data = data[:,1:]
    data = data.T
    data[data==0] = 1
    ori_data = data.copy()
    data = data[1:,:] / data[:-1,:]
    data = np.log(data)
    output = data[-duration:,:]
    output.shape = (1,duration,-1)
    return output, ori_data[-duration:,:]

def get_test3_data2(filename,duration):
    data = genfromtxt(filename, delimiter=',', skip_header=1)
    data = data[:,1:-1]
    data = data.T
    data[data==0] = 1
    ori_data = data.copy()
    data = -1*(data[1:,:] / data[:-1,:])    
    data = 1/(1+np.exp(data))
    output = data[-duration:,:]
    output.shape = (1,duration,-1)
    return output, ori_data[-duration:,:]

def get_test3_data2_1(filename,duration):
    data = genfromtxt(filename, delimiter=',', skip_header=1)
    data = data[:,1:]
    data = data.T
    data[data==0] = 1
    ori_data = data.copy()
    data = -1*(data[1:,:] / data[:-1,:])    
    data = 2/(1+np.exp(data))-1
    output = data[-duration:,:]
    output.shape = (1,duration,-1)
    return output, ori_data[-duration:,:]

def get_test3_data3(filename,duration,batch):
    data = genfromtxt(filename, delimiter=',', skip_header=1)
    data = data[:,1:-1]
    data = data.T
    data[data==0] = 1
    ori_data = data.copy()
    data = -1*(data[1:,:] / data[:-1,:])    
    data = 1/(1+np.exp(data))
    output = data[-duration:,batch]
    output.shape = (1,duration,-1)
    return output, ori_data[-duration:,batch]

def get_base_data(filename):
    data = genfromtxt(filename, delimiter=',', skip_header=1)
    data = data[:,1:]
    data = data.T
#    data[data==0] = 1
    ori_data = data.copy()
    return ori_data[-1,:]

