'''
构建cnn网络，分析mnist数据

'''

#--- 进阶版-----

import tensorflow as tf
import numpy as np 
import input_data
import AutoEncoder as AuEn

print('download and extract MNIST datasets')
mnist = input_data.read_data_sets('data/', one_hot=True)


# 加入自动编码器层
with tf.name_scope('AutoEncoder'):
    autoencoder=AuEn.AdditiveGaussianNoiseAutoencoder(
        n_input=784,n_hidden=200,transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)

    # 读取训练好的自动编码器：
    saver = tf.train.Saver()
    saver.restore(autoencoder.sess,'model/mnist_au_model.ckpt')

# 输入数据层
with tf.name_scope('Input'):
    x = tf.placeholder("float", shape=[None, 784],name='x_input')
    y_ = tf.placeholder("float", shape=[None, 10],name='y_input')



# 初始化偏置项
def weight_variable(shape):
  # 这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。
  # 和一般的正太分布的产生随机数据比起来，这个函数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。
  initial = tf.truncated_normal(shape, stddev=0.1,name='weight')
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape,name='bias')
  return tf.Variable(initial)

def conv2d(x, W):
  # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
  # W 是权重也是过滤器/内核张量
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 构建第一层卷积
# 把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。
with tf.name_scope('First_Layer'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # 把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数
    # (因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


with tf.name_scope('Second_Layer'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    # 构建第二层卷积
    # 为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch会得到64个特征。
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


with tf.name_scope('Full_acess'):
    # 现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。
    # 我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    # 我们通过tf.reshape()将h_pool2的输出值从一个三维的变为一维的数据, -1表示先不考虑输入图片例子维度, 将上一个输出结果展平.
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了减少过拟合，我们在输出层之前加入dropout
# 用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
# 这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。 
# TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。
# 所以用dropout的时候可以不用考虑scale。
with tf.name_scope('Drop_OUt'):
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 最后，我们添加一个softmax层，就像前面的单层softmax regression一样
# 输出层
with tf.name_scope('Output'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 训练和评估
with tf.name_scope('cross_entropy'):
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('Evaluate'):
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))




with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 输出summary，以便可视化
    summary_writer = tf.summary.FileWriter("cnn-logs/", sess.graph)

    for i in range(500):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        batch_xs_1 = autoencoder.reconstruct( batch_xs )
        #batch_xs_2 = autoencoder.reconstruct( batch_xs_1 )
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
            print ("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch_xs_1, y_: batch_ys, keep_prob: 1.0})
    
    print ("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))