# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 17:57:44 2018

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import pandas as pd

data = pd.read_excel('../ENB2012_data.xlsx', columns = ['X1', 'X2', 'X3', 'X4', 'X5',
                                                        'X6', 'X7', 'X8', 'Y1', 'Y2'])

#insize is features of input
# outsize: layer units for hidden layer; output features for output layer
#n is the No. n layer
def add_layer(inputs,insize,outsize,n,activation_function=None):
    layer_name='layer%s'%n
    with tf.name_scope(layer_name):
        Weights=tf.Variable(tf.random_normal([insize,outsize]),name='w')
        tf.summary.histogram(layer_name+'Weights',Weights)
        bias=tf.Variable(tf.zeros([1,outsize]))
        tf.summary.histogram(layer_name+'bias',bias)
        wx_b=tf.add(tf.matmul(inputs,Weights),bias)
        
        if activation_function is None:
            output=wx_b
        else:
            output=activation_function(wx_b,)
        return output
    
xdata = data.iloc[:, :8]#[] means the shape
#the input to feed_dict should be actual number
#instead of tensor
ydata = data.iloc[:, -2:]
xdata_norm = pd.DataFrame()
for i in xdata.columns:
    xdata_temp = 1.0 / (1 + np.exp(xdata[i]))
    xdata_norm = pd.concat([xdata_norm, xdata_temp], axis = 1)
xdata_norm

with tf.name_scope('input'):
   xs = tf.placeholder(tf.float32,xdata.shape,name='xinput')
   ys = tf.placeholder(tf.float32,ydata.shape,name='yinput')#target output

l1=add_layer(xs, xdata.shape[1], 10, 1, activation_function=tf.nn.relu)  #10 is layer units
l2=add_layer(l1, 10, 10, 2, activation_function=tf.nn.relu)  #10 is layer units
prediction=add_layer(l2, 10, 2, 3, activation_function=None) #predicted output

with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(prediction, ys)), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    optimizer=tf.train.GradientDescentOptimizer(0.001)
    train=optimizer.minimize(loss)

saver = tf.train.Saver()

with tf.Session() as sess:
   merged=tf.summary.merge_all()
   writer=tf.summary.FileWriter('log/',sess.graph)
   init=tf.global_variables_initializer()
   sess.run(init)
   for i in range(100000):
       sess.run(train,feed_dict={xs:xdata_norm,ys:ydata})
       if i%100==0:
           result=sess.run(merged,feed_dict={xs:xdata_norm, ys:ydata})
           print('第%s次预测结果：'% i, sess.run(prediction,feed_dict={xs:xdata_norm}), '\n')
           print(sess.run(loss,feed_dict={xs:xdata_norm, ys:ydata}))
           writer.add_summary(result,i)
   saver_path=saver.save(sess,'my_net/save_net.ckpt')          
   print ("save to path",saver_path)
#You must feed a value for placeholder tensor 'input_2/yinput' with dtype float and shape [200,1]
#if this erro happen just compile your tf in cmd
           
           
           

