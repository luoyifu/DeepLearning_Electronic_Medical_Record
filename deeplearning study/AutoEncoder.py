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

'''自编码器中会使用一种参数初始化方法xavier initialization，它的特点是会根据某一层网络的输入，输出节点数量自动调整最合适的分布。
如果深度学习模型的权重初始化得太小，那信号将在每层间传递时逐渐缩小而难以产生作用，但如果权重初始化得太大，
那信号将在每层间传递时逐渐放大并导致发散和失效。而Xaiver初始化器做的事情就是让权重被初始化得不大不小，正好合适。
即让权重满足0均值，同时方差为2／（n（in）+n(out)），分布可以用均匀分布或者高斯分布。
下面fan_in是输入节点的数量，fan_out是输出节点的数量。'''

def xavier_init(fan_in,fan_out,constant=1):
    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    high = constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    
    def __init__(self, 
                n_input, 
                n_hidden, 
                transfer_function =tf.nn.softplus, 
                optimizer = tf.train.AdamOptimizer(),  
                scale = 0.1):
        # scale参数表示噪声规模大小。构建hidden层时，给输入x增加了一个服从正态分布（高斯分布）的噪声
        # 噪声规模用scale修饰。未加规模参数的噪声取值[0,1]。
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # 定义网络结构，为输入x创建n_input长度的placeholder
        # 定义提取特征的隐藏层（hidden）：先将输入x加上噪声，然后用tf.matmul将加了噪声的输入与隐含层的权重相乘
        # 并使用tf.add加上隐含层的偏置，最后对结果进行激活函数处理。
        # 经过隐含层后，需要在输出层进行数据复原，重建操作(reconstruction)
        self.x=tf.placeholder(tf.float32,[None,self.n_input])
        
        self.hidden=self.transfer(tf.add(tf.matmul(self.x+scale * tf.random_normal((n_input,)),
                self.weights['w1']),self.weights['b1']))
        self.reconstruction=tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])
        
        # 接下来定义自编码器的损失函数，这里使用平方误差作为损失函数，
        # 再定义训练操作作为优化器对损失进行优化，最后创建Session并初始化自编码器的全部模型参数。
        self.cost=0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        self.optimizer=optimizer.minimize(self.cost)

        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)
    
    # 定义初始化参数的函数
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype= tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
        return all_weights

    # 定义计算损失cost及执行一步训练的函数partial_fit。
    # 函数里只需让Session执行两个计算图的节点，分别是损失cost和训练过程optimizer，输入的feed_dict包括输入数据x，
    # 以及噪声的系数scale。函数partial_fit做的就是用一个batch数据进行训练并返回当前的损失cost。
    def partial_fit(self,X):
        cost,opt=self.sess.run((self.cost,self.optimizer),feed_dict={self.x:X,self.scale:self.training_scale})
        return cost

    # 下面为一个只求损失cost的函数，这个函数是在自编码器训练完毕后，在测试集上对模型性能进行评测时会用到的。
    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})
    
    # 定义transform函数，返回自编码器隐含层的输出结果，
    # 它的目的是提供一个接口来获取抽象后的特征，自编码器的隐含层的最主要功能就是学习出数据中的高阶特征。
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})
    
    # 定义generate函数，将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据
    def generate(self,hidden=None):
        if hidden is None:
            hidden=np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})
    
    # 定义reconstruct函数，它整体运行一遍复原过程，包括提取高阶特征和通过高阶特征复原数据
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.training_scale})

    # 定义getWeights函数的作用是获取隐含层的权重w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    # 定义getBiases函数则是获取隐含层的偏置系数b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])
