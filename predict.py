# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 15:46:12 2018

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import random

tf.reset_default_graph()

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

#xdata = []
#for i in range(8):
#    xdata.append(random.random())
#xdata = np.array(xdata).reshape([1, 8])
xdata = [0.98, 514.50, 294.00, 110.25, 7.00, 2, 0.10, 1]
xdata = np.array(xdata).reshape([1, 8])
xdata = 1.0 / (1 + np.exp(xdata))
print(xdata)
    
xs = tf.placeholder(tf.float32, xdata.shape, name='xinput')

l1=add_layer(xs, xdata.shape[1], 10, 1, activation_function=tf.nn.relu)  #10 is layer units
l2=add_layer(l1, 10, 10, 2, activation_function=tf.nn.relu)  #10 is layer units
prediction=add_layer(l2, 10, 2, 3, activation_function=None) #predicted output

saver = tf.train.Saver()

with tf.Session() as sess:
    # you cannot initialize here
   saver.restore(sess,'my_net/save_net.ckpt')
   print(sess.run(prediction,feed_dict={xs:xdata}))