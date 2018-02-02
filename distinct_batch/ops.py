# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:16:05 2017

@author: charlie
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util

def _weights(name, shape, mean=0.0, stddev=0.02):
  var = tf.get_variable(
    name, shape,
    initializer=tf.random_normal_initializer(mean=mean, stddev=stddev, dtype=tf.float32))
    #initializer=tf.contrib.layers.xavier_initializer())
#    var = tf.get_variable(name,
#                          shape,
#                          initializer=tf.contrib.layers.xavier_initializer())
  return var
  
def _biases(name, shape, constant=0.0):
  return tf.get_variable(name, shape,
            initializer=tf.constant_initializer(constant))
            
def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        batch_size = value.get_shape()[0].value
        duration = value.get_shape()[1].value
        channels = value.get_shape()[2].value
        pad_elements = dilation - 1 - (duration + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, channels])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        output = tf.reshape(transposed, [batch_size * dilation, -1, channels])
        return output

def batch_to_time(value, dilation, name=None):
    with tf.name_scope('batch_to_time'):
        batch_size = value.get_shape()[0].value
        duration = value.get_shape()[1].value
        channels = value.get_shape()[2].value
        new_batch_size = batch_size//dilation
        prepared = tf.reshape(value, [dilation, -1, channels])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,
                          [new_batch_size, -1, channels])

def _instance_norm(input):
  with tf.variable_scope("instance_norm"):
    depth = input.get_shape()[3]
    scale = _weights("scale", [depth], mean=1.0)
    offset = _biases("offset", [depth])
    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset
    
def _bn(input):
    depth = input.get_shape()[3]
    beta = _biases("offset", [depth])
    gamma = _weights("scale", [depth], mean=1.0)
    mean, variance = tf.nn.moments(input, axes=[0,1,2])
    output = tf.nn.batch_normalization(input, mean, variance, beta, gamma, 0.01)
    return output

def _make_output(temp_output,norm,activation):
    if norm == 'instance':
        normalized = _instance_norm(temp_output)
    if norm == 'batch':
        normalized = _bn(temp_output)
    if norm == None:
        normalized = temp_output
    if activation == 'relu':
        output = tf.nn.relu(normalized)
    if activation == 'tanh':
        output = tf.nn.tanh(normalized)
    if activation == 'sigmoid':
        output = tf.nn.sigmoid(normalized)
    if activation == 'elu':
        output = tf.nn.elu(normalized)
    if activation == None:
        output = normalized
    return output
         
def causal_residue(input,depth,dilation,kernel_size,norm,activation,scope):
    with tf.variable_scope(scope):
        x = causal_conv(input,depth,dilation,kernel_size,norm,activation,'dilation')
        input_cut = input.get_shape()[1].value - x.get_shape()[1].value
        input = tf.slice(input, [0, input_cut, 0], [-1, -1, -1])
        return x + input

def causal_side(input,depth,dilation,kernel_size,norm,activation,scope):
    with tf.variable_scope(scope):
        x = causal_conv(input,depth,dilation,kernel_size,norm,activation,'dilation')
        input_cut = input.get_shape()[1].value - x.get_shape()[1].value
        input = tf.slice(input, [0, input_cut, 0], [-1, -1, -1])
        cut = causal_conv(x,depth,1,1,norm,activation,'side')
        return x + input, cut

def combine_cut(side_cut,layer_number,depth):
    final_scope = 'cc'+str(layer_number-1)
    final_cut = side_cut[final_scope]
    cut_length = final_cut.get_shape()[1].value
    for i in range(layer_number-1):
        scope = 'cc'+str(i)
        cut = side_cut[scope]
        ori_length = cut.get_shape()[1].value
        cut = tf.slice(cut, [0, ori_length-cut_length, 0], [-1, -1, -1])
        final_cut = final_cut + cut
    output = causal_conv(final_cut,depth,1,1,None,'relu','post')
    output = causal_conv(output,1,1,1,None,None,'output')
    return output
    
def combine_cut2(side_cut,layer_number,depth):
    final_scope = 'cc'+str(layer_number-1)
    final_cut = side_cut[final_scope]
    cut_length = final_cut.get_shape()[1].value
    for i in range(layer_number-1):
        scope = 'cc'+str(i)
        cut = side_cut[scope]
        ori_length = cut.get_shape()[1].value
        cut = tf.slice(cut, [0, ori_length-cut_length, 0], [-1, -1, -1])
        final_cut = tf.concat(2,[final_cut,cut])
    output = causal_conv(final_cut,depth,1,1,None,'relu','post')
    output = causal_conv(output,1,1,1,None,None,'output')
    return output

def causal_gate(input,depth,dilation,kernel_size,norm,activation,scope):
    with tf.variable_scope(scope):
        x = causal_conv(input,depth,dilation,kernel_size,norm,activation,'dilation')
        x_gate = causal_conv(x,depth,1,kernel_size,norm,'sigmoid','gate')
        x_filter = causal_conv(x,depth,1,kernel_size,norm,'tanh','filter')
        x = x_filter * x_gate
        x = causal_conv(x,depth,1,1,norm,activation,'post')
        input_cut = input.get_shape()[1].value - x.get_shape()[1].value
        input = tf.slice(input, [0, input_cut, 0], [-1, -1, -1])
        cut = causal_conv(x,depth,1,1,norm,activation,'side')
        return x+input,cut
                 
def causal_conv(input,
                output_dim,
                dilation,
                kernel_size=2,
                norm=None,
                activation='tanh',
                scope='causal_conv'):
    with tf.variable_scope(scope):
        weights = _weights("weights",
                           shape=[kernel_size, input.get_shape()[2], output_dim])
        biases = _biases("biases", [output_dim])
        if dilation > 1:
            transformed = time_to_batch(input, dilation)
            conv = tf.nn.conv1d(transformed, weights, stride=1, padding='VALID')
            conv += biases
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(input, weights, stride=1, padding='VALID')
            restored += biases
        out_width = input.get_shape()[1].value - (kernel_size - 1) * dilation
        result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, out_width, -1])
        result = result[:,None,:,:]
        result = _make_output(result,norm,activation)
        result = result[:,0,:,:]
        return result  

def normal_conv(input,
                output_dim,
                kernel_size,
                stride,
                norm=None,
                activation='tanh',
                scope='conv',
                padding='VALID'):
    with tf.variable_scope(scope):
        weights = _weights("weights",
                           shape=[kernel_size,input.get_shape()[2],output_dim])
        biases = _biases("biases",[output_dim])
        conv = tf.nn.conv1d(input,weights,stride=stride,padding=padding)
        output = conv + biases
        output = output[:,None,:,:]
        output = _make_output(output,norm,activation)
        output = output[:,0,:,:]
        return output

def normal_residue(input_,norm,activation,scope):
    with tf.variable_scope(scope):
        output_dim = input_.get_shape()[2].value
        cut = input_
        x = normal_conv(input_,output_dim,1,1,norm,activation,'r1','SAME')
        x = normal_conv(x,output_dim,1,1,norm,activation,'r2','SAME')
        return x + cut
        
def normal_gate(input_,norm,scope):
    with tf.variable_scope(scope):
        output_dim = input_.get_shape()[2].value
        cut = input_
        process = normal_conv(input_,output_dim,3,1,norm,'tanh','r1','SAME')
        selection = normal_conv(input_,output_dim,3,1,norm,'sigmoid','r2','SAME')
        return process*selection + cut

def gc_conv(input,output_dim,scope='conv'):
    with tf.variable_scope(scope):
        weights = _weights("weights",
                           shape=[1,input.get_shape()[2],output_dim])
        conv = tf.nn.conv1d(input,weights,stride=1,padding='SAME')
        return conv

def normal_dconv(input,
                 output_dim,
                 kernel_size,
                 stride,
                 norm=None,
                 activation='relu',
                 scope='dconv',
                 padding='SAME'):
    with tf.variable_scope(scope):
        input = input[:,None,:,:]
        weights = _weights("weights",
                           shape=[1,kernel_size,output_dim,input.get_shape()[3].value])
        biases = _biases("biases",[output_dim])
        output_size = input.get_shape()[2].value*stride
        output_shape = [input.get_shape()[0].value, 1, output_size, output_dim]
        conv = tf.nn.conv2d_transpose(input, weights,
                                            output_shape=output_shape,
                                            strides=[1, 1, stride, 1], padding=padding)
        output = conv + biases
        output = _make_output(output,norm,activation)
        output = output[:,0,:,:]
        return output
        
def dconv_gate(input,
               output_dim,
               kernel_size,
               stride,
               norm=None,
               activation='relu',
               scope='dconv',
               padding='SAME'):
    with tf.variable_scope(scope):
        process = normal_dconv(input,output_dim,kernel_size,stride,norm,'tanh','process',padding)
        selection = normal_dconv(input,output_dim,kernel_size,stride,norm,'sigmoid','selection',padding)
        output = process * selection
        output = normal_conv(output,output_dim,kernel_size,1,norm,activation,'output',padding)        
        return output

#def normal_dconv(input,
#                 output_dim,
#                 kernel_size,
#                 stride,
#                 norm=None,
#                 activation='relu',
#                 scope='dconv',
#                 padding='SAME'):
#    with tf.variable_scope(scope):
#        input = input[:,None,:,:]
#        duration = input.get_shape()[2].value
#        input = tf.image.resize_images(input,(1,duration*2),method=0)
#        input = input[:,0,:,:]
#        output = normal_conv(input,output_dim,3,1,norm,activation,'r1','SAME')
#        return output
        
#def subpixel(input,stride,norm,activation,scope):
#    batch_size = input.get_shape()[0].value
#    duration = input.get_shape()[1].value
#    output_dim = input.get_shape()[2].value
#    with tf.variable_scope(scope):
#        x = normal_conv(input,output_dim,1,1,norm,activation,'pre_sub') #[b,d,32]
#        x = tf.reshape(x,[batch_size,duration,stride,output_dim//stride]) #[b,d,2,16]
#        x = tf.split(1,duration,x) #d*[b,1,2,16]
#        x = tf.concat(1,[n[:,0,:,:] for n in x]) #[b,2*d,16]
#        return x

def fold(x,keep_channel=False):
    batch_size = x.get_shape()[0].value
    output_dim = x.get_shape()[2].value
    if keep_channel:
#        x = tf.reshape(x,[-1,2,output_dim])
#        x = tf.reshape(x,[-1,1,output_dim*2])
#        x = tf.transpose(x,[1,0,2])
#        x = tf.transpose(x,[0,2,1])
#        x = x[:,None,:,:]
#        x = tf.nn.avg_pool(x,[1,1,2,1],[1,1,2,1],'SAME')
#        x = x[:,0,:,:]
#        x = tf.transpose(x,[0,2,1])
         x = x[:,None,:,:]
         x = tf.reshape(x,[batch_size,-1,2,output_dim])
         x = tf.reshape(x,[batch_size,-1,1,output_dim*2])
         x = tf.transpose(x,[0,2,1,3])
         x = tf.transpose(x,[0,1,3,2])
         x = tf.nn.avg_pool(x,[1,1,2,1],[1,1,2,1],'SAME')
         x = x[:,0,:,:]
         x = tf.transpose(x,[0,2,1])
    else:
#        x = tf.reshape(x,[-1,2,output_dim])
#        x = tf.reshape(x,[-1,1,output_dim*2])
#        x = tf.transpose(x,[1,0,2])
         x = x[:,None,:,:]
         x = tf.reshape(x,[batch_size,-1,2,output_dim])
         x = tf.reshape(x,[batch_size,-1,1,output_dim*2])
         x = tf.transpose(x,[0,2,1,3])
         x = x[:,0,:,:]
    return x

def unfold(x,keep_channel=False):
    batch_size = x.get_shape()[0].value
    output_dim = x.get_shape()[2].value
    x = x[:,None,:,:]
    x = tf.transpose(x,[0,2,1,3])
    x = tf.reshape(x,[batch_size,-1,2,output_dim//2])
    x = tf.reshape(x,[batch_size,1,-1,output_dim//2])
    x = x[:,0,:,:]
    if keep_channel:
        #x = tf.reshape(x,[batch_size,-1,output_dim//2])
        x = x[:,None,:,:]
        x = tf.transpose(x,[0,1,3,2])
        x = tf.image.resize_images(x,(1,output_dim),method=0)
        x = tf.transpose(x,[0,1,3,2])
        x = x[:,0,:,:]
#    else:
#        x = tf.reshape(x,[batch_size,-1,output_dim//2])
    return x

#def down_lstm(x,state,channel_limit,norm,act,scope):
#    batch_size = x.get_shape()[0].value
#    output_dim = x.get_shape()[2].value
#    with tf.variable_scope(scope):
#        if output_dim == channel_limit:
#            output_channel = output_dim
#            keep_channel = True
#        else:
#            output_channel = output_dim * 2
#            keep_channel = False
#        selection1 = normal_conv(x,output_dim,3,1,norm,'sigmoid','selection1','SAME')
#        state = state * selection1
#        process2 = normal_conv(x,output_dim,3,1,norm,act,'process2','SAME')
#        selection2 = normal_conv(x,output_dim,3,1,norm,'sigmoid','selection2','SAME')
#        state = state + process2 * selection2
#        selection3 = normal_conv(x,output_channel,3,2,norm,'sigmoid','selection3','SAME')
#        process3 = normal_conv(state,output_channel,3,2,norm,act,'process3','SAME')
#        x = selection3 * process3
#        state = fold(state,keep_channel)
#        return x, state

def down_fold(x,channel_limit,norm,act,scope):
    output_dim = x.get_shape()[2].value
    if output_dim == channel_limit:
        keep_channel = True
    else:
        keep_channel = False
    x = fold(x,keep_channel)
    return x
    
def up_fold(x,upper_num,norm,act,scope):
    output_dim = x.get_shape()[2].value
    if output_dim == upper_num:
        keep_channel = True
    else:
        keep_channel = False
    x = unfold(x,keep_channel)
    return x

def down_seperation(x,state,channel_limit,norm,act,scope):
    batch_size = x.get_shape()[0].value
    output_dim = x.get_shape()[2].value
    state_dim = state.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == channel_limit:
            keep_channel = True
        else:
            keep_channel = False
        selection1 = normal_conv(state,output_dim,3,1,norm,'sigmoid','selection1','SAME')
        selected = x * (1-selection1)
        x = x * selection1
        process2 = normal_conv(selected,state_dim,3,1,norm,act,'process2','SAME')
        selectoin2 = normal_conv(selected,state_dim,3,1,norm,'sigmoid','selection2','SAME')
        state = state + process2*selectoin2
        state = fold(state,True)
        x = fold(x,keep_channel)
        return x,state

def down_conv2(x,channel_limit,norm,act,scope):
    batch_size = x.get_shape()[0].value
    output_dim = x.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == channel_limit:
            keep_channel = True
        else:
            keep_channel = False
        cut = x
        x = normal_conv(x,output_dim,3,1,norm,act,'conv0','SAME')
        x = normal_conv(x,output_dim,3,1,norm,act,'conv1','SAME')
        x = x + cut
        x = fold(x,keep_channel)
        return x
        
def res_conv(x,norm,act,scope):
    output_dim = x.get_shape()[2].value
    with tf.variable_scope(scope):
        #cut = x
        #output_dim = output_dim // 2
        x = normal_conv(x,output_dim,3,1,norm,act,'conv0','SAME')
        x = normal_conv(x,output_dim*2,3,1,norm,act,'conv1','SAME')
        x = x[:,None,:,:]
        x = tf.nn.max_pool(x,[1,1,2,1],[1,1,2,1],'SAME')
        x = x[:,0,:,:]
        #x = x + cut
        return x
 
def down_conv_pool(x,channel_limit,norm,act,scope):
    batch_size = x.get_shape()[0].value
    output_dim = x.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == channel_limit:
            cut = x
        else:
            output_dim = output_dim * 2
            cut = x
            cut = cut[:,None,:,:]
            cut = tf.transpose(cut,[0,1,3,2])
            cut = tf.image.resize_images(cut,(1,output_dim),method=0)
            cut = tf.transpose(cut,[0,1,3,2])
            cut = cut[:,0,:,:]
        x = normal_conv(x,output_dim,3,1,norm,act,'conv0','SAME')
        x = normal_conv(x,output_dim,3,1,norm,act,'conv1','SAME')
        x = cut + x
        x = x[:,None,:,:]
        x = tf.nn.max_pool(x,[1,1,2,1],[1,1,2,1],'SAME')
        x = x[:,0,:,:]
        return x

def pool2d(input_):
    input_ = input_[:,None,:,:]
    input_ = tf.nn.max_pool(input_,[1,1,2,1],[1,1,2,1],'SAME')
    return input_[:,0,:,:]

#def down_conv_pool(x,channel_limit,norm,act,scope):
#    batch_size = x.get_shape()[0].value
#    output_dim = x.get_shape()[2].value
#    with tf.variable_scope(scope):
#        if output_dim == channel_limit:
#            cut = x
#            cut = pool2d(cut)
#        else:
#            output_dim = output_dim * 2
#            cut = x
#            cut = cut[:,None,:,:]
#            cut = tf.transpose(cut,[0,1,3,2])
#            cut = tf.image.resize_images(cut,(1,output_dim),method=0)
#            cut = tf.transpose(cut,[0,1,3,2])
#            cut = tf.nn.max_pool(cut,[1,1,2,1],[1,1,2,1],'SAME')
#            cut = cut[:,0,:,:]
#        x = normal_conv(x,output_dim,3,1,norm,act,'conv0','SAME')
#        x = normal_conv(x,output_dim,3,1,norm,act,'conv1','SAME')
#        x = pool2d(x)
#        return x  + cut       
       
def down_gate2(x,channel_limit,norm,act,scope):
    batch_size = x.get_shape()[0].value
    output_dim = x.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == channel_limit:
            keep_channel = True
        else:
            keep_channel = False
        cut = x
        selection = normal_conv(x,output_dim,3,1,norm,'sigmoid','selection','SAME')
        x = x * selection
        process = normal_conv(x,output_dim,3,1,norm,act,'process','SAME')
        x = cut + process # 原为x = x + selection*process
        x = fold(x,keep_channel)
        return x
        
def down_gate2_valid(x,channel_limit,norm,act,scope):
    batch_size = x.get_shape()[0].value
    output_dim = x.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == channel_limit:
            keep_channel = True
        else:
            keep_channel = False
        cut = x
        selection = normal_conv(x,output_dim,3,1,norm,'sigmoid','selection','VALID')
        process = normal_conv(x,output_dim,3,1,norm,act,'process','VALID')
        cut_num = cut.get_shape()[1].value - selection.get_shape()[1].value
        cut = tf.slice(cut, [0, cut_num, 0], [-1, -1, -1])
        x = cut + selection * process
        if x.get_shape()[1].value % 2 == 1:
            x = x[:,1:,:]
        #print (x)
        x = fold(x,keep_channel)
        return x
                    
def down_gate2_c(x,gc_embedding,channel_limit,norm,act,scope):
    batch_size = x.get_shape()[0].value
    output_dim = x.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == channel_limit:
            keep_channel = True
        else:
            keep_channel = False
        p_value = gc_conv(gc_embedding,output_dim,'pconv')
        s_value = gc_conv(gc_embedding,output_dim,'sconv')
        cut = x
        selection = normal_conv(x,output_dim,3,1,norm,None,'selection','SAME')
        selection = tf.nn.sigmoid(selection+s_value)
        x = x * selection
        process = normal_conv(x,output_dim,3,1,norm,None,'process','SAME')
        process = tf.nn.tanh(process+p_value)
        x = cut + process # 原为x = x + selection*process
        x = fold(x,keep_channel)
        return x
        
def up_gate2(x,upper_num,norm,act,scope):
    batch_size = x.get_shape()[0].value
    output_dim = x.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == upper_num:
            keep_channel = True
        else:
            keep_channel = False
        cut = x
        selection = normal_conv(x,output_dim,3,1,norm,'sigmoid','selection','SAME')
        process = normal_conv(x,output_dim,3,1,norm,act,'process','SAME')
        x = x + selection * process
        x = unfold(x,keep_channel)
        return x
            
def up_conv2(x,upper_num,norm,act,scope):
    batch_size = x.get_shape()[0].value
    output_dim = x.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == upper_num:
            keep_channel = True
        else:
            keep_channel = False
        cut = x
        x = normal_conv(x,output_dim,3,1,norm,act,'conv0','SAME')
        x = normal_conv(x,output_dim,3,1,norm,act,'conv1','SAME')
        x = x + cut
        x = unfold(x,keep_channel)
        return x

def attach_condition(input_,category,cat_dim):
    duration = input_.get_shape()[1].value
    label_tensor = tf.zeros([1,duration,cat_dim],dtype=tf.float32)
    x = tf.one_hot(category,cat_dim,dtype=tf.float32)
    x = tf.reshape(x,[1,1,-1])
    label_tensor += x
    output = tf.concat(2,[input_,label_tensor])
    return output

def down_seperation_c(x,state,gc_embedding,channel_limit,norm,act,scope):
    batch_size = x.get_shape()[0].value
    output_dim = x.get_shape()[2].value
    state_dim = state.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == channel_limit:
            keep_channel = True
        else:
            keep_channel = False
        selection1 = normal_conv(state,output_dim,3,1,norm,'sigmoid','selection1','SAME')
        selected = x * (1-selection1)
        x = x * selection1
        #selected = attach_condition(selected,category,cat_dim)
        p_value = gc_conv(gc_embedding,state_dim,'pconv')
        s_value = gc_conv(gc_embedding,state_dim,'sconv')
        process2 = normal_conv(selected,state_dim,3,1,norm,None,'process2','SAME')
        process2 = tf.tanh(process2+p_value)
        selectoin2 = normal_conv(selected,state_dim,3,1,norm,None,'selection2','SAME')
        selectoin2 = tf.sigmoid(selectoin2+s_value)
        state = state + process2*selectoin2
        state = fold(state,True)
        x = fold(x,keep_channel)
        return x,state
        
def up_seperation(x,state,upper_num,norm,act,scope):
    batch_size = x.get_shape()[0].value
    output_dim = x.get_shape()[2].value
    state_dim = state.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == upper_num:
            keep_channel = True
        else:
            keep_channel = False
        selection1 = normal_conv(x,state_dim,3,1,norm,'sigmoid','selection1','SAME')
        selected = state * (1-selection1)
        state = state * selection1
        process2 = normal_conv(selected,output_dim,3,1,norm,act,'process2','SAME')
        selection2 = normal_conv(selected,output_dim,3,1,norm,'sigmoid','selection2','SAME')
        x = x + process2 * selection2
        state = unfold(state,True)
        x = unfold(x,keep_channel)
        return x, state

def up_seperation_c(x,state,gc_embedding,upper_num,norm,act,scope):
    batch_size = x.get_shape()[0].value
    output_dim = x.get_shape()[2].value
    state_dim = state.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == upper_num:
            keep_channel = True
        else:
            keep_channel = False
        selection1 = normal_conv(x,state_dim,3,1,norm,'sigmoid','selection1','SAME')
        selected = state * (1-selection1)
        state = state * selection1
        #selected = attach_condition(selected,category,cat_dim)
        p_value = gc_conv(gc_embedding,output_dim,'pconv')
        s_value = gc_conv(gc_embedding,output_dim,'sconv')
        process2 = normal_conv(selected,output_dim,3,1,norm,None,'process2','SAME')
        process2 = tf.tanh(process2+p_value)
        selection2 = normal_conv(selected,output_dim,3,1,norm,None,'selection2','SAME')
        selection2 = tf.sigmoid(selection2+s_value)
        x = x + process2 * selection2
        state = unfold(state,True)
        x = unfold(x,keep_channel)
        return x, state
        

def down_lstm(x,state,channel_limit,norm,act,scope):
    batch_size = x.get_shape()[0].value
    output_dim = x.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == channel_limit:
            keep_channel = True
        else:
            keep_channel = False
        selection1 = normal_conv(x,output_dim,3,1,norm,'sigmoid','selection1','SAME')
        state = state + x
        state = state * selection1
        
        process2 = normal_conv(x,output_dim,3,1,norm,act,'process2','SAME')
        selection2 = normal_conv(x,output_dim,3,1,norm,'sigmoid','selection2','SAME')
        state = state + process2 * selection2
        selection3 = normal_conv(x,output_dim,3,1,norm,'sigmoid','selection3','SAME')
        process3 = normal_conv(state,output_dim,3,1,norm,act,'process3','SAME')
        x = selection3 * process3
        x = fold(x,keep_channel)
        state = fold(state,keep_channel)
        return x, state

def up_lstm(x,state,upper_num,norm,act,scope):
    batch_size = x.get_shape()[0].value
    output_dim = x.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == upper_num:
            keep_channel = True
        else:
            keep_channel = False
        selection1 = normal_conv(x,output_dim,3,1,norm,'sigmoid','selection1','SAME')
        state = state + x
        state = state * selection1
        
        process2 = normal_conv(x,output_dim,3,1,norm,act,'process2','SAME')
        selection2 = normal_conv(x,output_dim,3,1,norm,'sigmoid','selection2','SAME')
        state = state + process2 * selection2
        selection3 = normal_conv(x,output_dim,3,1,norm,'sigmoid','selection3','SAME')
        process3 = normal_conv(state,output_dim,3,1,norm,act,'process3','SAME')
        x = selection3 * process3
        x = unfold(x,keep_channel)
        state = unfold(state,keep_channel)
        return x, state
       
#def up_lstm(x,state,upper_num,norm,act,scope):
#    batch_size = x.get_shape()[0].value
#    output_dim = x.get_shape()[2].value
#    with tf.variable_scope(scope):
#        if output_dim == upper_num:
#            output_channel = output_dim * 2
#            keep_channel = True
#        else:
#            output_channel = output_dim
#            keep_channel = False
#        selection1 = normal_conv(x,output_dim,3,1,norm,'sigmoid','selection1','SAME')
#        state = state * selection1
#        process2 = normal_conv(x,output_dim,3,1,norm,act,'process2','SAME')
#        selection2 = normal_conv(x,output_dim,3,1,norm,'sigmoid','selection2','SAME')
#        state = state + process2 * selection2
#        selection3 = normal_conv(x,output_channel,3,1,norm,'sigmoid','selection3','SAME')
#        process3 = normal_conv(state,output_channel,3,1,norm,act,'process3','SAME')
#        x = selection3 * process3
#        x = unfold(x,False)
#        state = unfold(state,keep_channel)
#        return x, state
            

def up_gate(input,upper_num,norm,act,scope):
    batch_size = input.get_shape()[0].value
    output_dim = input.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == upper_num:
            x = tf.reshape(input,[batch_size,-1,output_dim//2])
            x = tf.transpose(x,[0,2,1])
            x = tf.image.resize_images(x,(1,output_dim),method=0)
            x = tf.transpose(x,[0,2,1])
            cut = x
            x_process = normal_conv(x,output_dim*2,3,2,norm,act,'process','SAME')
            x_selection = normal_conv(x,output_dim*2,3,2,norm,'sigmoid','selection','SAME')
            x = x_process * x_selection
#            x_process = normal_conv(x,output_dim*2,3,2,norm,act,'process1','SAME')
#            x_selection = normal_conv(x,output_dim*2,3,2,norm,'sigmoid','selection1','SAME')
#            x = x_process * x_selection
            x = tf.reshape(x,[batch_size,-1,output_dim])
            output = x + cut
        else:
            x = tf.reshape(input,[batch_size,-1,output_dim//2])
            cut = x
            x_process = normal_conv(x,output_dim,3,2,norm,act,'process','SAME')
            x_selection = normal_conv(x,output_dim,3,2,norm,'sigmoid','selection','SAME')
            x = x_process * x_selection
#            x_process = normal_conv(x,output_dim,3,2,norm,act,'process1','SAME')
#            x_selection = normal_conv(x,output_dim,3,2,norm,'sigmoid','selection1','SAME')
#            x = x_process * x_selection
            x = tf.reshape(x,[batch_size,-1,output_dim//2])
            output = x + cut
        return output


def down_gate(input,norm,act,scope):
    batch_size = input.get_shape()[0].value
    output_dim = input.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == 128:
            cut = tf.reshape(input,[batch_size,-1,output_dim*2])
            cut = tf.transpose(cut,[0,2,1])
            cut = cut[:,None,:,:]
            cut = tf.nn.max_pool(cut,[1,1,2,1],[1,1,2,1],'SAME')
            cut = cut[:,0,:,:]
            cut = tf.transpose(cut,[0,2,1])
            x_process = normal_conv(input,output_dim,3,2,norm,act,'process','SAME')
            x_selection = normal_conv(input,output_dim,3,2,norm,'sigmoid','selection','SAME')
            x = x_process * x_selection
#            x_process = normal_conv(x,output_dim,3,2,norm,act,'process1','SAME')
#            x_selection = normal_conv(x,output_dim,3,2,norm,'sigmoid','selection1','SAME')
#            x = x_process * x_selection
            output = x + cut
        else:
            cut = tf.reshape(input,[batch_size,-1,output_dim*2])
            x_process = normal_conv(input,output_dim*2,3,2,norm,act,'process','SAME')
            x_selection = normal_conv(input,output_dim*2,3,2,norm,'sigmoid','selection','SAME')
#            x = x_process * x_selection
#            x_process = normal_conv(x,output_dim*2,3,2,norm,act,'process1','SAME')
#            x_selection = normal_conv(x,output_dim*2,3,2,norm,'sigmoid','selection1','SAME')
            output = x_process*x_selection + cut
        return output

def down_gate_c(input,gc_embedding,norm,act,scope):
    batch_size = input.get_shape()[0].value
    output_dim = input.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == 512:
            cut = tf.reshape(input,[batch_size,-1,output_dim*2])
            cut = tf.transpose(cut,[0,2,1])
            cut = cut[:,None,:,:]
            cut = tf.nn.max_pool(cut,[1,1,2,1],[1,1,2,1],'SAME')
            cut = cut[:,0,:,:]
            cut = tf.transpose(cut,[0,2,1])
            p_value = gc_conv(gc_embedding,output_dim,'pconv')
            s_value = gc_conv(gc_embedding,output_dim,'sconv')
            x_process = normal_conv(input,output_dim,3,2,norm,act,'process','SAME')
#            x_process = x_process + p_value
#            x_process = tf.tanh(x_process)
            x_selection = normal_conv(input,output_dim,3,2,norm,'sigmoid','selection','SAME')
#            x_selection = x_selection + s_value
#            x_selection = tf.sigmoid(x_selection)
            x = x_process * x_selection
            output = x + cut
        else:
            cut = tf.reshape(input,[batch_size,-1,output_dim*2])
            p_value = gc_conv(gc_embedding,output_dim*2,'pconv')
            s_value = gc_conv(gc_embedding,output_dim*2,'sconv')
            x_process = normal_conv(input,output_dim*2,3,2,norm,act,'process','SAME')
#            x_process = x_process + p_value
#            x_process = tf.tanh(x_process)
            x_selection = normal_conv(input,output_dim*2,3,2,norm,'sigmoid','selection','SAME')
#            x_selection = x_selection + s_value
#            x_selection = tf.sigmoid(x_selection)
            output = x_process*x_selection + cut
        return output
            
def up_gate_c(input,gc_embedding,upper_num,norm,act,scope):
    batch_size = input.get_shape()[0].value
    output_dim = input.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == upper_num:
            x = tf.reshape(input,[batch_size,-1,output_dim//2])
            x = tf.transpose(x,[0,2,1])
            x = tf.image.resize_images(x,(1,output_dim),method=0)
            x = tf.transpose(x,[0,2,1])
            cut = x
            p_value = gc_conv(gc_embedding,output_dim*2,'pconv')
            s_value = gc_conv(gc_embedding,output_dim*2,'sconv')
            x_process = normal_conv(x,output_dim*2,3,2,norm,None,'process','SAME')
            x_process = x_process + p_value
            x_process = tf.tanh(x_process)
            x_selection = normal_conv(x,output_dim*2,3,2,norm,None,'selection','SAME')
            x_selection = x_selection + s_value
            x_selection = tf.sigmoid(x_selection)
            x = x_process * x_selection
            x = tf.reshape(x,[batch_size,-1,output_dim])
            output = x + cut
        else:
            x = tf.reshape(input,[batch_size,-1,output_dim//2])
            cut = x
            p_value = gc_conv(gc_embedding,output_dim,'pconv')
            s_value = gc_conv(gc_embedding,output_dim,'sconv')
            x_process = normal_conv(x,output_dim,3,2,norm,None,'process','SAME')
            x_process = x_process + p_value
            x_process = tf.tanh(x_process)
            x_selection = normal_conv(x,output_dim,3,2,norm,None,'selection','SAME')
            x_selection = x_selection + s_value
            x_selection = tf.sigmoid(x_selection)
            x = x_process * x_selection
            x = tf.reshape(x,[batch_size,-1,output_dim//2])
            output = x + cut
        return output            
            

def down_conv(input,channel_limit,norm,act,scope):
    batch_size = input.get_shape()[0].value
    output_dim = input.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == channel_limit:
            cut = tf.reshape(input,[batch_size,-1,output_dim*2])
            cut = tf.transpose(cut,[0,2,1])
            cut = cut[:,None,:,:]
            cut = tf.nn.max_pool(cut,[1,1,2,1],[1,1,2,1],'SAME')
            cut = cut[:,0,:,:]
            cut = tf.transpose(cut,[0,2,1])
            x = normal_conv(input,output_dim,3,2,norm,act,'shuffle0','SAME')
#            x = tf.reshape(x,[batch_size,-1,output_dim//2])
#            x = normal_conv(x,output_dim,3,2,norm,act,'shuffle1','SAME')
            output = x + cut
        else:
            cut = tf.reshape(input,[batch_size,-1,output_dim*2])
            x = normal_conv(input,output_dim*2,3,2,norm,act,'shuffle0','SAME')
#            x = tf.reshape(x,[batch_size,-1,output_dim])
#            x = normal_conv(x,output_dim*2,3,2,norm,act,'shuffle1','SAME')
            output = x + cut
        return output
        
#def up_conv(input,norm,act,scope):
#    batch_size = input.get_shape()[0].value
#    output_dim = input.get_shape()[2].value
#    with tf.variable_scope(scope):
#        x = tf.reshape(input,[batch_size,-1,output_dim//2])
#        cut = x
#        x = normal_conv(x,output_dim,3,2,norm,act,'shuffle0','SAME')
#        x = tf.reshape(x,[batch_size,-1,output_dim//2])
#        x = normal_conv(x,output_dim,3,2,norm,act,'shuffle1','SAME')
#        x = tf.reshape(x,[batch_size,-1,output_dim//2])
#        return x + cut



def up_conv(input,upper_num,norm,act,scope):
    batch_size = input.get_shape()[0].value
    output_dim = input.get_shape()[2].value
    with tf.variable_scope(scope):
        if output_dim == upper_num:
            x = tf.reshape(input,[batch_size,-1,output_dim//2])
            x = tf.transpose(x,[0,2,1])
            x = tf.image.resize_images(x,(1,output_dim),method=0)
            x = tf.transpose(x,[0,2,1])
            cut = x
            x = normal_conv(x,output_dim*2,3,2,norm,act,'shuffle0','SAME')
            x = tf.reshape(x,[batch_size,-1,output_dim])
#            x = normal_conv(x,output_dim*2,3,2,norm,act,'shuffle1','SAME')
#            x = tf.reshape(x,[batch_size,-1,output_dim])
            output = x + cut
        else:
            x = tf.reshape(input,[batch_size,-1,output_dim//2])
            cut = x
            x = normal_conv(x,output_dim,3,2,norm,act,'shuffle0','SAME')
            x = tf.reshape(x,[batch_size,-1,output_dim//2])
#            x = normal_conv(x,output_dim,3,2,norm,act,'shuffle1','SAME')
#            x = tf.reshape(x,[batch_size,-1,output_dim//2])
            output = x + cut
        return output


def suppixel(input,stride,norm,activation,scope):
    batch_size = input.get_shape()[0].value
    duration = input.get_shape()[1].value
    output_dim = input.get_shape()[2].value
    with tf.variable_scope(scope):
        cut = tf.reshape(input,[batch_size,-1,output_dim*2])
        x = normal_conv(input,output_dim*2,stride,stride,norm,activation,'shuffle0') #shrink
        x = tf.reshape(x,[batch_size,-1,output_dim]) #expand
        x = normal_conv(x,output_dim*2,stride,stride,norm,activation,'shuffle1') #shrink
        return x + cut
        
        
def subpixel(input,stride,norm,activation,scope):
    batch_size = input.get_shape()[0].value
    duration = input.get_shape()[1].value
    output_dim = input.get_shape()[2].value
    with tf.variable_scope(scope):
        x = tf.reshape(input,[batch_size,-1,output_dim//stride]) #expand
        cut = x
        x = normal_conv(x,output_dim,stride,stride,norm,activation,'shuffle0') #shrink
        x = tf.reshape(x,[batch_size,-1,output_dim//stride]) #expand
        x = normal_conv(x,output_dim,stride,stride,norm,activation,'shuffle1') #shrink
        x = tf.reshape(x,[batch_size,-1,output_dim//stride]) #expand
        return x + cut

def fc(input,
       output_dim,
       activation='relu',
       name='fcblock'):
    with tf.variable_scope(name):
        input_shape = input.get_shape().as_list()
        weights = _weights("weights",
                           shape=[input_shape[1],output_dim])                 
        #biases = _biases("biases",[output_dim])
        weights_square = weights * weights
        weights_s_sum = tf.reduce_sum(weights_square,reduction_indices=0,keep_dims=True)
        weights_sqrt = tf.sqrt(weights_s_sum)
        weights = weights / weights_sqrt
        temp_output = tf.matmul(input, weights) #+ biases
        return _make_output(temp_output,None,activation)

def graph_saver(output_node,output_name,sess):
    output_node_ = tf.mul(output_node,1.0,name='output')
    output_node_name = 'output'
    output_graph_path = output_name
    graph_def = sess.graph.as_graph_def()
    s_graph_def = graph_util.remove_training_nodes(graph_def)
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, s_graph_def, output_node_name.split(","))
    with tf.gfile.GFile(output_graph_path, "wb") as f:
        f.write(output_graph_def.SerializeToString())

def graph_loader(input_node,pb_path):
    with tf.gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, {'input':input_node})
    graph = tf.get_default_graph()
    output_node = 'import/output:0'
    output = graph.get_tensor_by_name(output_node)
    return output







