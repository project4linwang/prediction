# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:26:57 2018

@author: charlie
"""

import tensorflow.python as tf
import numpy as np
from ops import fc, causal_conv

def net(x,n_hidden,n_layer,reuse=False):
    with tf.variable_scope('net',reuse=reuse):
        #得到输入数据维度
        batch_size = x.get_shape()[0].value
        duration = x.get_shape()[1].value
        channels = x.get_shape()[2].value
        #定义一个LSTM cell
        cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
        if not reuse:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=0.4)
        #定义多层LSTM cell
        cells = tf.nn.rnn_cell.MultiRNNCell([cell]*n_layer,state_is_tuple=True)
        #得到初始状态
        initial_state = cells.zero_state(batch_size, tf.float32)
        #将输入数据做一个矩阵乘，将不同通道的数据混合在一起
        inputs = tf.reshape(x,[-1,channels])
        inputs = fc(inputs,n_hidden,None,'fc')
        inputs = tf.reshape(inputs,[batch_size,duration,-1])
        #将输入数据放入lstm中，得到输出序列和最终状态
        output, last_state = tf.nn.dynamic_rnn(cells,inputs,initial_state=initial_state)
        #将输出序列重新映射为和输入数据一致的通道数
        output = tf.reshape(output,[-1,n_hidden])
        output = fc(output,channels,None,'fc2')
        output = tf.reshape(output,[batch_size,duration,-1])
        return output, initial_state, last_state
        
def wavenet(input,depth,reuse=False,scope='net'):
    with tf.variable_scope(scope,reuse=reuse):
        x = causal_conv(input,depth,1,2,scope='cc1',activation='tanh')
        for i in range(5):
            x = causal_conv(x,depth,2**i,2,None,'tanh','cc'+str(i+2))
        x = causal_conv(x,666,1,2,scope='output',activation=None)
        return x