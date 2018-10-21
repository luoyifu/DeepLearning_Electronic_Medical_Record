
import tensorflow as tf
import numpy as np 
import input_data
import AutoEncoder as AuEn

print('download and extract MNIST datasets')
mnist = input_data.read_data_sets('data/', one_hot=True)
print ('number of train data is %d' % (mnist.train.num_examples))
print ('number of test data is %d' % (mnist.test.num_examples))


# ---初级版----

# ---定义输入（input）数据---


'''
# 希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784]
  x = tf.placeholder("float", [None, 784])
# 定义新的占位符y_ 用来输入真实值
  y_ = tf.placeholder("float", [None,10])
'''
with tf.name_scope('AutoEncoder'):
  autoencoder=AuEn.AdditiveGaussianNoiseAutoencoder(
    n_input=784,n_hidden=200,transfer_function=tf.nn.softplus,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)

  saver = tf.train.Saver()
  
  saver.restore(autoencoder.sess,'model/mnist_au_model.ckpt')

  x_au = tf.placeholder("float",[None, 784], name='x_au_in')
  y_au_ = tf.placeholder("float",[None, 10], name='y_au_in')


with tf.name_scope('inputs'):
# 使用with tf.name_scope('inputs')可以将xs和ys包含进来，形成一个大的图层，图层的名字就是with tf.name_scope()方法里的参数。
# 输入数据为了可视化进行一下更改
  x = tf.placeholder("float", [None,784], name='x_in')
  y_ = tf.placeholder("float", [None, 10], name='y_in')


with tf.name_scope('softmax'):
  # 用全为零的张量来初始化W和b。因为我们要学习W和b的值，它们的初值可以随意设置。
  W = tf.Variable(tf.zeros([784,10]), name='weight')
  b = tf.Variable(tf.zeros([10]), name='bias')
  # 用tf.matmul(​​X，W)表示x乘以W
  y = tf.nn.softmax(tf.matmul(x_au ,W) + b)


# 定义交叉熵
with tf.name_scope('cross_entropy'):
  cross_entropy = -tf.reduce_sum(y_au_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 在训练的过程在参数是不断地在改变和优化的，我们往往想知道每次迭代后参数都做了哪些变化
# 可以将参数的信息展现在tenorbord上，因此我们专门写一个方法来收录每次的参数信息。
# 定义一个方法：
def variable_summaries(var):
    # Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    with tf.name_scope('summaries'):
      # 计算参数的均值，并使用tf.summary.scaler记录
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)

      # 计算参数的标准差
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      # 用直方图记录参数的分布
      tf.summary.histogram('histogram', var)


#- 模型构建完毕，开始模型部署--

# 初始化变量
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

sess = tf.Session()

summary_writer = tf.summary.FileWriter("logs/", sess.graph)
# 将所有summary节点合并成一个节点
# merged_summary_op = tf.summary.merge_all()

sess.run(init)

# 这里我们让模型循环训练1000次
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  batch_xs_au = autoencoder.reconstruct(batch_xs)
  sess.run(train_step, feed_dict={x_au: batch_xs_au, y_au_: batch_ys})
  #if i % 100 == 0:
  #  summary_str = sess.run(merged_summary_op)
  #  summary_writer.add_summary(summary_str, total_step)

with tf.name_scope('evaluate'):
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_au_,1))
  # 把布尔值转换成浮点数，然后取平均值。例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
  # tf.cast(x,dtype)，将x类型转换为dtype
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print (sess.run(accuracy, feed_dict={x_au: mnist.test.images, y_au_: mnist.test.labels}))
