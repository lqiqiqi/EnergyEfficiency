from django.shortcuts import render

# Create your views here.

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 15:46:12 2018

@author: Administrator
"""
from django.shortcuts import render, render_to_response
from django.views.decorators import csrf
import tensorflow as tf
import numpy as np


def add_layer(inputs, insize, outsize, n, activation_function=None):
    layer_name = 'layer%s' % n
    with tf.name_scope(layer_name):
        Weights = tf.Variable(tf.random_normal([insize, outsize]), name='w')
        tf.summary.histogram(layer_name + 'Weights', Weights)
        bias = tf.Variable(tf.zeros([1, outsize]))
        tf.summary.histogram(layer_name + 'bias', bias)
        wx_b = tf.add(tf.matmul(inputs, Weights), bias)

        if activation_function is None:
            output = wx_b
        else:
            output = activation_function(wx_b, )
        return output

def input(request):
    return render_to_response('get.html')

def calculate(request):
    # xdata = [0.98, 514.50, 294.00, 110.25, 7.00, 2, 0.10, 1]
    request.encoding = 'utf-8'
    context = {}
    xdata = []
    if request.GET:
        for num in range(1, 9):
            xdata.append(float(request.GET['X%i' % num]))

    xdata = np.array(xdata).reshape([1, 8])
    xdata = 1.0 / (1 + np.exp(xdata))
    # print('xdata: ', xdata)
    # context['heatload'] = xdata[0]
    # context['coolload'] = xdata[2]
    xs = tf.placeholder(tf.float32, xdata.shape, name='xinput')

    l1 = add_layer(xs, xdata.shape[1], 10, 1, activation_function=tf.nn.relu)  # 10 is layer units
    l2 = add_layer(l1, 10, 10, 2, activation_function=tf.nn.relu)  # 10 is layer units
    prediction = add_layer(l2, 10, 2, 3, activation_function=None)  # predicted output

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # you cannot initialize here
        saver.restore(sess, './my_net/save_net.ckpt')
        rlt = sess.run(prediction, feed_dict={xs: xdata})
        # print('result: ', rlt)
        context['heatload'] = rlt[0, 0]
        context['coolload'] = rlt[0, 1]
        print('context', context)
    return render(request, "result.html", context)