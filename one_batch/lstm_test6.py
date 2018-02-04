# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 10:58:11 2018

@author: charlie
"""

#使用log(Xn/Xn-1)作为数据的归一化

import tensorflow.python as tf
import numpy as np
import matplotlib.pyplot as plt
from model import net
from input_data2 import make_lstm4_batch, get_test2_data

def make_test():
    prime, ori_data = get_test2_data(filename,duration)
    temp_state = sess.run(test_istate)
    for i in range(ori_data.shape[0]):
        x = prime[:,i,:]
        x.shape = (1,) + x.shape
        feed = {test_holder:x,test_istate:temp_state}
        temp_state,temp_output = sess.run([test_lstate,test_output],feed)
    test_result = np.zeros((test_datas,666))
    for i in range(test_datas):
        x = temp_output
        feed = {test_holder:x,test_istate:temp_state}
        temp_state,temp_output = sess.run([test_lstate,test_output],feed)
        test_result[i,:] = temp_output[0,0,:]
    result = np.zeros_like(test_result)
    result[0,:] = ori_data[-1,:]
    for i in range(0,test_datas-1):
        temp_result = test_result[i,:]
        result[i+1,:] = result[i,:] * np.exp(temp_result+1)
        result[result<0] = 0
    test_target = result[1,:]
    test_loss = np.sqrt(np.mean((test_target-input_target)**2))
    return test_loss

tf.reset_default_graph() #reset一下比较好
sess = tf.Session() #开一个新session
filename = './data/data_six.csv' #使用的原始训练数据
duration = 10 #每次训练使用的序列长度
n_hidden = 64 #lstm神经元个数
n_layer = 1 #lstm层数
test_datas = 70 #预测数据点个数


#制作输入训练数据队列
input_data,input_label,input_target = make_lstm4_batch(filename,duration,False) 
#测试数据输入的place holder
test_holder = tf.placeholder(tf.float32,[1,1,666]) 
#通过模型得到输出
output,_,_ = net(input_data,n_hidden,n_layer) 
#通过模型得到测试输出，测试初始状态，测试最终状态
test_output,test_istate,test_lstate = net(test_holder,n_hidden,n_layer,True) 
#定义损失函数，是输出和输入的差方
loss_op = tf.sqrt(tf.reduce_mean(tf.squared_difference(output,input_label)))
#定义训练，使用Adam优化器
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss_op)
#初始化所有变量
sess.run(tf.initialize_all_variables())
#开始运行输入队列
tf.train.start_queue_runners(sess=sess)
#开始训练循环
scores = []
for step in range(1000000):
    _,loss_value = sess.run([train_op,loss_op])
    score = 100.0
    if step % 10 == 0:
        score = make_test()*666/140
        print (score,step)
        scores.append(score)
    if score < 50.0:
        break
    #print (loss_value)
plt.plot(scores)
#得到测试初始化数据的归一化数据和原数据   
prime, ori_data = get_test2_data(filename,duration)
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

#plt.plot(test_result)

#以下代码块为将以上得到的预测数据还原为归一化之前的数据
result = np.zeros_like(test_result)
result[0,:] = ori_data[-1,:]

for i in range(0,test_datas-1):
    temp_result = test_result[i,:]
    result[i+1,:] = result[i,:] * np.exp(temp_result+1)
    result[result<0] = 0

test_target = result[1,:]
test_loss = np.sqrt(np.mean((test_target-input_target)**2))
print (test_loss)

result = np.sum(result,1)


plt.figure()
plt.plot(result)

print(result[1])

np.savetxt("result.csv", result[1:3,:].T, delimiter=",")






