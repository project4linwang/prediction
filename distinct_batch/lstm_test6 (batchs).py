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
from input_data2_1 import  make_lstm5_batch3, get_test3_data3, get_base_data
import util

filename = './data/data_six_fill.csv' #使用的原始训练数据
duration = 20#每次训练使用的序列长度
n_hidden = 512
#lstm神经元个数
n_layer = 2 #lstm层数
test_datas = 3 #预测数据点个数
channels=1
total_cars=666
n_result=[]
total=[]
class_id=140

for batch in range(total_cars):
    tf.reset_default_graph() #reset一下比较好
    sess = tf.Session() #开一个新session
    
    #制作输入训练数据队列
    #input_data,input_label = make_lstm5_batch(filename,duration,False) 
    input_data,input_label = make_lstm5_batch3(filename,duration,batch,False)
    
    
    #测试数据输入的place holder
    test_holder = tf.placeholder(tf.float32,[1,1,channels]) 
#    print test_holder
    #通过模型得到输出
    output,_,_ = net(input_data,n_hidden,n_layer) 
    #通过模型得到测试输出，测试初始状态，测试最终状态
    test_output,test_istate,test_lstate = net(test_holder,n_hidden,n_layer,True) 
    #定义损失函数，是输出和输入的差方
    loss_op = tf.reduce_mean((output-input_label)**2)
    #定义训练，使用Adam优化器
    train_op = tf.train.AdamOptimizer(0.0001).minimize(loss_op)
    #初始化所有变量
    sess.run(tf.global_variables_initializer())
    #开始运行输入队列
    tf.train.start_queue_runners(sess=sess)
    #开始训练循环
    
#    base_data=get_base_data(filename)
    min_score=100
    for step in range(1000):
        _,loss_value = sess.run([train_op,loss_op])
        #print (loss_value)
        if step % 200 == 100:
            prime, ori_data = get_test3_data3(filename,duration,batch)
            temp_state = sess.run(test_istate)
            for i in range(ori_data.shape[0]):
                x = prime[:,i,:]
                x.shape = (1,) + x.shape
                #将输入数据和初始状态放到place holder中
                feed = {test_holder:x,test_istate:temp_state}
                #运行模型，得到最终状态和最终输出，并赋值给当前状态和当前输出
                temp_state,temp_output = sess.run([test_lstate,test_output],feed)
            test_result = np.zeros((test_datas,channels))
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
            result[0] = ori_data[-1]
            total_score=0
            for i in range(1,test_datas):
    #            temp_result = test_result[i,:]
    #            temp_result = np.exp(temp_result)
    #            result[i,:] = result[i-1,:] * temp_result
    #            result[result<0] = 0
                temp_result = test_result[i]
                temp_result = np.log(1/temp_result-1)
                result[i] = result[i-1] * temp_result*(-1)
                result[result<0] = 0
                
#            for i in range(0,666):
#                diff=math.pow((result[1][i]-base_data[i]),2)
#                total_score=total_score+diff
                
#            total_score=math.sqrt(total_score/666)   
                
            #output 1:201710 ; 2:201711
#            print(total_score,n_result[1],n_result[2],loss_value)
            print('Batch:',batch,' loss:',loss_value)
            print result[1],result[2]
            #record min level
    #        if total_score<min_score and loss_value<0.0001 and step>200:
    #            min_score=total_score    
    #            min_result=result
    #            min_n_result=n_result
    #            min_test_result=test_result
    #            print(total_score,n_result[1],n_result[2],loss_value)
                
    
#    plt.plot(test_result)
    n_result.append([result[1],result[2]])
    print 'close batch',batch
    sess.close()
    
base_data=get_base_data(filename)
base_data=util.from_666_to_140(filename,base_data)
listtemp=[]
listout=[]
for item in n_result:
    listtemp.append(item[0])
    listout.append(item[1])
listIDK=util.from_666_to_140(filename,listtemp)
output=util.from_666_to_140(filename,listout)
util.to_csv('multi_bat',output)

for i in range(0,class_id):
    diff=math.pow((listIDK[i]-base_data[i]),2)
    total_score=total_score+diff
            
total_score=math.sqrt(total_score/140)    
total1=0
total2=0
#total=np.sum(n_result,0)
for i in range(0,total_cars):
    total1=total1+n_result[i][0]
    total2=total2+n_result[i][1]
print total_score,total1,total2
plt.figure()
plt.plot(output)