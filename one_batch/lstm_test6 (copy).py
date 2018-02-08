# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 10:58:11 2018

@author: charlie
"""

#使用log(Xn/Xn-1)作为数据的归一化
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from model import net
from input_data2_1 import make_lstm5_batch2_1, get_test3_data2_1, get_base_data
import util

tf.reset_default_graph() #reset一下比较好
sess = tf.Session() #开一个新session
filename = './data/data_six_clip.csv' #使用的原始训练数据
duration =20 #每次训练使用的序列长度
n_hidden = 64 #lstm神经元个数
n_layer = 2 #lstm层数
test_datas = 3 #预测数据点个数
class_id=140
fileN='s-'+str(duration)+'-'+str(n_hidden)+'-c'
#制作输入训练数据队列
#input_data,input_label = make_lstm5_batch(filename,duration,False) 
input_data,input_label = make_lstm5_batch2_1(filename,duration,False)

#666 batches
#trans_input = tf.transpose(input_data,[2,1,0])
#trans_label = tf.transpose(input_label,[2,1,0])
#input_data=trans_input
#input_label=trans_label

#测试数据输入的place holder
test_holder = tf.placeholder(tf.float32,[1,1,666]) 
#通过模型得到输出
output,_,_ = net(input_data,n_hidden,n_layer) 
#通过模型得到测试输出，测试初始状态，测试最终状态
test_output,test_istate,test_lstate = net(test_holder,n_hidden,n_layer,True) 
#定义损失函数，是输出和输入的差方
loss_op = tf.reduce_mean(tf.abs(output-input_label))
#定义训练，使用Adam优化器
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss_op)
#初始化所有变量
sess.run(tf.global_variables_initializer())
#开始运行输入队列
tf.train.start_queue_runners(sess=sess)
#开始训练循环

base_data=get_base_data(filename)
base_data=util.from_666_to_140(filename,base_data)
min_score=100
losses=[]
for step in range(50100):
    _,loss_value = sess.run([train_op,loss_op])
    #print (loss_value)
    if step % 100 == 99:
        prime, ori_data = get_test3_data2_1(filename,duration)
        temp_state = sess.run(test_istate)
        for i in range(ori_data.shape[0]):
            x = prime[:,i,:]
            x.shape = (1,) + x.shape
            #将输入数据和初始状态放到place holder中
            feed = {test_holder:x,test_istate:temp_state}
            #运行模型，得到最终状态和最终输出，并赋值给当前状态和当前输出
            temp_state,temp_output = sess.run([test_lstate,test_output],feed)
        test_result = np.zeros((test_datas,666))
        for i in range(test_datas):
            x = temp_output
            #将测试初始化数据的最终数据和最终状态输入到模型中
            feed = {test_holder:x,test_istate:temp_state}
            #运行模型得到下一个预测数据和最终状态
            temp_state,temp_output = sess.run([test_lstate,test_output],feed)
            test_result[i,:] = temp_output[0,0,:]
#        if step==0:
#            print test_result
        result = np.zeros_like(test_result)
        result[0,:] = ori_data[-1,:]
        total_score=0
        for i in range(1,test_datas):
#            temp_result = test_result[i,:]
#            temp_result = np.exp(temp_result)
#            result[i,:] = result[i-1,:] * temp_result
#            result[result<0] = 0
            temp_result = test_result[i,:]
            temp_result = np.log(2/(temp_result+1)-1)
            result[i,:] = result[i-1,:] * temp_result*(-1)
            result[result<0] = 0
        
#        print np.sum(base_data,0)
        listIDK=util.from_666_to_140(filename,result[1])
        for i in range(0,class_id):
            diff=math.pow((listIDK[i]-base_data[i]),2)
            total_score=total_score+diff
            
#        print total_score
#        total_score=math.sqrt(total_score/class_id)            
        n_result = np.sum(result,1)
        #output 1:201710 ; 2:201711
        print(n_result[1],loss_value)
        losses.append(loss_value)
        #record min level
#        if total_score<min_score :
#            min_score=total_score    
#            min_result=result
#            min_n_result=n_result
#            min_test_result=test_result
#            print(total_score,n_result[1],n_result[2],loss_value)
            
'''
#得到测试初始化数据的归一化数据和原数据   
prime, ori_data = get_test3_data2(filename,duration)
#prime = prime[:,:-1,:]
#ori_data = ori_data[:-1,:]

#下面代码块为得到测试初始化数据的最终状态
#先得到初始状态
temp_state = sess.run(test_istate)
for i in range(ori_data.shape[0]):
    x = prime[:,i,:]
    x.shape = (1,) + x.shape
    #将输入数据和初始状态放到place holder中
    feed = {test_holder:x,test_istate:temp_state}
    #运行模型，得到最终状态和最终输出，并赋值给当前状态和当前输出
    temp_state,temp_output = sess.run([test_lstate,test_output],feed)

#让test_result存储模型预测数据
test_result = np.zeros((test_datas,666))
for i in range(test_datas):
    x = temp_output
    #将测试初始化数据的最终数据和最终状态输入到模型中
    feed = {test_holder:x,test_istate:temp_state}
    #运行模型得到下一个预测数据和最终状态
    temp_state,temp_output = sess.run([test_lstate,test_output],feed)
    test_result[i,:] = temp_output[0,0,:]

plt.plot(test_result)

#以下代码块为将以上得到的预测数据还原为归一化之前的数据
result = np.zeros_like(test_result)
result[0,:] = ori_data[-1,:]

for i in range(1,test_datas):
#    temp_result = test_result[i,:]
#    temp_result = np.exp(temp_result)
#    result[i,:] = result[i-1,:] * temp_result
#    result[result<0] = 0
    temp_result = test_result[i,:]
    temp_result = np.log(1/temp_result-1)
    result[i,:] = result[i-1,:] * temp_result*(-1)
    result[result<0] = 0    

result = np.sum(result,1)
'''
plt.figure()
plt.plot(listIDK)
plt.figure()
plt.plot(losses)
util.to_csv(fileN,listIDK)
#print(result[1])

