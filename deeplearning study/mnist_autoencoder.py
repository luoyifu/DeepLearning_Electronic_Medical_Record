'''
实验给mnist的深度学习加入了自动编码器
自动编码器相关理论知识可以参考
https://luoyifu.github.io/luoyifu.github.io/2018/06/11/AutoEncoder_tensorflow/
'''

'''
#一种解决方案
#首先，导入将要使用到的各种库和数据集，定义各个参数如学习率、训练迭代次数等，清晰明了便于后期修改。
#由于自编码器的神经网络结构非常有规律性，都是xW + b的结构，故将每一层的权重W和偏置b的变量tf.Variable统一置于一个字典中
#通过字典的key值更加清晰明了的描述。模型构建思路上，将编码器部分和解码器部分分开构建，每一层的激活函数使用Sigmoid函数
#编码器通常与编码器使用同样的激活函数。通常编码器部分和解码器部分是一个互逆的过程，例如我们设计将784维降至256维再降至128维的编码器
#解码器对应的就是从128维解码至256维再解码至784维。定义代价函数，代价函数表示为解码器的输出与原始输入的最小二乘法表达
#优化器采用AdamOptimizer训练阶段每次循环将所有的训练数据都参与训练。经过训练，最终将训练结果与原数据可视化进行对照。
#如果增大训练循环次数或者增加自编码器的层数，可以得到更好的还原效果。
# 参考https://blog.csdn.net/marsjhao/article/details/68950697 

import tensorflow as tf  
import numpy as np  
import matplotlib.pyplot as plt  
  
# 导入MNIST数据  
from tensorflow.examples.tutorials.mnist import input_data  
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)  
  
learning_rate = 0.01  
training_epochs = 10  
batch_size = 256  
display_step = 1  
examples_to_show = 10  
n_input = 784  
  
# tf Graph input (only pictures)  
X = tf.placeholder("float", [None, n_input])  
  
# 用字典的方式存储各隐藏层的参数  
n_hidden_1 = 256 # 第一编码层神经元个数  
n_hidden_2 = 128 # 第二编码层神经元个数  
# 权重和偏置的变化在编码层和解码层顺序是相逆的  
# 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数 

#--- 为何这样设置权重？---
# 参考https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-11-autoencoder/
# 在压缩环节：我们要把这个Features不断压缩，经过第一个隐藏层压缩至256个 Features，再经过第二个隐藏层压缩至128个。
# 在解压环节：我们将128个Features还原至256个，再经过一步还原至784个。
# 在对比环节：比较原始数据与还原后的拥有 784 Features 的数据进行 cost 的对比，
# 根据 cost 来提升我的 Autoencoder 的准确率，下图是两个隐藏层的 weights 和 biases 的定义： 
weights = {  
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),  
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),  
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),  
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),  
}  
biases = {  
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),  
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),  
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),  
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),  
}  
  
# 每一层结构都是 xW + b  
# 构建编码器  
def encoder(x):  
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),  
                                   biases['encoder_b1']))  
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),  
                                   biases['encoder_b2']))  
    return layer_2  
  
  
# 构建解码器  
def decoder(x):  
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),  
                                   biases['decoder_b1']))  
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),  
                                   biases['decoder_b2']))  
    return layer_2  
  
# 构建模型  
encoder_op = encoder(X)  
decoder_op = decoder(encoder_op)  
  
# 预测  
y_pred = decoder_op  
y_true = X  
  
# 定义代价函数和优化器  
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) #最小二乘法  
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)  

# 决定是否进行训练，还是从文件中读取训练结果
is_Trained = True

with tf.Session() as sess:  
    # tf.initialize_all_variables() no long valid from  
    # 2017-03-02 if using tensorflow >= 0.12  

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  
        init = tf.initialize_all_variables()  
    else:  
        init = tf.global_variables_initializer()  
    sess.run(init)  

    saver = tf.train.Saver()
    if is_Trained:
        saver.restore(sess,'model/model.ckpt')
    else:
    # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练  
        total_batch = int(mnist.train.num_examples/batch_size) #总批数  
        for epoch in range(training_epochs):  
            for i in range(total_batch):  
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0  
                # Run optimization op (backprop) and cost op (to get loss value)  
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})  
            if epoch % display_step == 0:  
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))  
        print("Optimization Finished!")  
        saver.save(sess,'model/model.ckpt')

    encode_decode = sess.run(  
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})  
    f, a = plt.subplots(2, 10, figsize=(10, 2))  
    for i in range(examples_to_show):  
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))  
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))  
    plt.show()  

'''

#----------------------------------------------------------------#
# --降噪自动编码器
# 参考 https://blog.csdn.net/qq_31531635/article/details/76158288
# 定义降噪自动编码器类

# 实现一个去噪自编码器和实现一个单隐含层的神经网络差不多，
# 只不过是在数据输入时做了标准化，并加上了一个高斯噪声，同时我们的输出结果不是数字分类结果，
# 而是复原的数据，因此不需要用标注过的数据进行监督训练。
# 自编码器作为一种无监督学习的方法，它与其它无监督学习的主要不同在于，它不是对数据进行聚类，
# 而是提取其中最有用，最频繁出现的高阶特征，根据这些高阶特征重构数据。
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt  

import sklearn.preprocessing as prep

import AutoEncoder as AuEn

from tensorflow.examples.tutorials.mnist import input_data  

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  


'''自编码器中会使用一种参数初始化方法xavier initialization，它的特点是会根据某一层网络的输入，输出节点数量自动调整最合适的分布。
如果深度学习模型的权重初始化得太小，那信号将在每层间传递时逐渐缩小而难以产生作用，但如果权重初始化得太大，
那信号将在每层间传递时逐渐放大并导致发散和失效。而Xaiver初始化器做的事情就是让权重被初始化得不大不小，正好合适。
即让权重满足0均值，同时方差为2／（n（in）+n(out)），分布可以用均匀分布或者高斯分布。
下面fan_in是输入节点的数量，fan_out是输出节点的数量。'''

# 定义一个对训练、测试数据进行标准化处理的函数
# 标准化即让数据变成0均值且标准差为1的分布。方法就是先减去均值，再除以标准差。
def standard_scale(X_train,X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train=preprocessor.transform(X_train)
    X_test=preprocessor.transform(X_test)
    return X_train,X_test

# 再定义一个获取随机block数据的函数：
# 取一个从0到len(data)-batch_size之间的随机整数，
# 再以这个随机数作为block的起始位置，然后顺序取到一个batch size的数据。
# 要注意的是，这属于不放回抽样，可以提高数据的利用效率
def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0,len(data) - batch_size)
    return data[start_index:(start_index+batch_size)]

# 用之前定义的standard_scale函数对训练集、测试机进行标准化变换


# 创建一个自编码器的实例，定义模型输入节点数n_input为784，
# 自编码器的隐含层点数n_hidden为200，隐含层的激活函数transfer_function为softplus，优化器optimizer为Adam
# 且学习速率为0。001，同时将噪声的系数设为0.01
with tf.name_scope('AutoEncoder'):
    autoencoder=AuEn.AdditiveGaussianNoiseAutoencoder(
        n_input=784,n_hidden=196,transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)

# 定义是重新训练模型还是读取已有模型，如果is_Trained = True， 则表示模型已经训练过了，直接读取模型
is_Trained = True
saver = tf.train.Saver()

if is_Trained :
    saver.restore(autoencoder.sess,'model/mnist_au_model.ckpt')
else:
    X_train,X_test=standard_scale(mnist.train.images,mnist.test.images)
    # 下面定义几个常用参数，总训练样本数，最大训练的轮数(traning_epochs)设为20，
    # batch_size设为128，并设置每隔一轮(epoch)就显示一次损失cost
    # 下面开始训练过程，在每一轮(epoch)循环开始时，将平均损失avg_cost设为0，
    # 并计算总共需要的batch数（通过样本总数除以batch大小），
    # 在每一轮迭代后，显示当前的迭代数和这一轮迭代的平均cost。
    n_samples=int(mnist.train.num_examples)
    training_epochs=31
    batch_size=128
    display_step=1
    examples_to_show = 10 
    for epoch in range(training_epochs):
        total_batch = int(n_samples/batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)
            cost = autoencoder.partial_fit(batch_xs)
        if epoch%5 == 0:
            print ("step %d, training cost %g"%(epoch, cost))

    saver.save(autoencoder.sess,'model/mnist_au_model.ckpt') 
print('The model is ready!')   

'''
au=[]
au = autoencoder.reconstruct(mnist.test.images[:examples_to_show])

f, a = plt.subplots(2, 10, figsize=(10, 2))  
for i in range(examples_to_show):  
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))  
    a[1][i].imshow(np.reshape(au[i], (28, 28)))  
plt.show() 
'''


with tf.name_scope('inputs'):
# 使用with tf.name_scope('inputs')可以将xs和ys包含进来，形成一个大的图层，图层的名字就是with tf.name_scope()方法里的参数。
# 输入数据为了可视化进行一下更改
  # x = tf.placeholder("float", [None,784], name='x_in')
  y_ = tf.placeholder("float", [None, 10], name='y_in')
  x_h = tf.placeholder("float", [None, 196], name='x_hidden_in')


with tf.name_scope('softmax'):
  # 用全为零的张量来初始化W和b。因为我们要学习W和b的值，它们的初值可以随意设置。
  W = tf.Variable(tf.zeros([196,10]), name='weight')
  b = tf.Variable(tf.zeros([10]), name='bias')
  # 用tf.matmul(​​X，W)表示x乘以W
  y = tf.nn.softmax(tf.matmul(x_h,W) + b)

# 定义交叉熵
with tf.name_scope('cross_entropy'):
  cross_entropy = -tf.reduce_sum(y_*tf.log(y))

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

summary_writer = tf.summary.FileWriter("mnist_logs/", sess.graph)
# 将所有summary节点合并成一个节点
# merged_summary_op = tf.summary.merge_all()

sess.run(init)

# 这里我们让模型循环训练1000次
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  batch_xs_h = autoencoder.transform( batch_xs )
  sess.run(train_step, feed_dict={x_h: batch_xs_h, y_: batch_ys})
  if i % 100 == 0:
      print("step %d"%i)
  #  summary_str = sess.run(merged_summary_op)
  #  summary_writer.add_summary(summary_str, total_step)

with tf.name_scope('evaluate'):
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  # 把布尔值转换成浮点数，然后取平均值。例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
  # tf.cast(x,dtype)，将x类型转换为dtype
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

mti_h = autoencoder.transform(mnist.test.images) 
yp = sess.run(y, feed_dict={x_h:mti_h})
cn = sess.run(correct_prediction, feed_dict={y:yp,y_:mnist.test.labels})
ac = tf.reduce_mean(tf.cast(cn,"float"))
print (sess.run(accuracy, feed_dict={x_h: mti_h , y_: mnist.test.labels}))
