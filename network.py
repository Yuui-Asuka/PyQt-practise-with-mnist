# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PyQt5.QtCore import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

#每一层的单元数
tf.flags.DEFINE_integer("hidden_cells",500,"number of cells each layer.(default:500)")
#分类数
tf.flags.DEFINE_integer("n_classes",10,"number of classes.(default:10)")
#权重矩阵标准差
tf.flags.DEFINE_float("stddev",0.1,"standard deviation of weight matrix.(default:0.1)")
#偏置
tf.flags.DEFINE_float("bias",0.1,"bias of each layer.(default:0.1)")
#学习率
tf.flags.DEFINE_float("lr",0.001,"Learning rate.(default:0.001)")
#dropout参数
tf.flags.DEFINE_float("dropout",1.0,"dropout keep prob.(default:1.0)")
# 批次大小
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 50)")
# 迭代周期
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 50)")
#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()


class Worker(QThread):
    breakSignal = pyqtSignal(str)
    stopSignal = pyqtSignal(name = 'stop')

    def __init__(self,hidden_cells,epoch,learning_rate,layer_num,optimizer,activation_function,
                                  loss_function,nomorlization,batch_size,keep_prob,bias,stddev,clip,batch_norm,decay):
        super().__init__()   
        self.hidden_cells = hidden_cells
        self.epoch = epoch
        self.learning_rate = learning_rate      
        self.layer_num = layer_num
        self.optimizer = optimizer
        self.activation = activation_function
        self.loss = loss_function
        self.nomorlization = nomorlization
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.bias = bias
        self.stddev = stddev
        self.clip = clip
        self.batch_norm = batch_norm
        self.decay = decay

    def batch_norm_layer(self,value,is_training=False,name='batch_norm'):
        if not isinstance (self.decay,float):
            self.decay = 0.95  
        if is_training is True:
            return tf.contrib.layers.batch_norm(inputs=value,decay=self.decay,updates_collections=None,is_training = True)
        else:
            return tf.contrib.layers.batch_norm(inputs=value,decay=self.decay,updates_collections=None,is_training = False)
        
    def run(self):
        if not isinstance(self.learning_rate,float) or self.learning_rate == 0:
            self.learning_rate = 0.001
        if not isinstance(self.layer_num,int) or self.layer_num == 0:
            self.layer_num = 3
        if not isinstance(self.batch_size,int) or self.batch_size == 0:
            self.batch_size = 50
        if not isinstance(self.keep_prob,(int,float)) or self.keep_prob == 0:
            self.keep_prob = 1.0
        if not isinstance(self.bias,(int,float)):
            self.bias = 0.1
        if not isinstance(self.stddev,(int,float)):
            self.stddev = 0.1        
        if not isinstance (self.epoch,int) or self.epoch == 0:
            self.epoch = 1
        self.sess = tf.InteractiveSession()
        batch_size = self.batch_size
        hidden_cells = self.hidden_cells
        n_batch = mnist.train.num_examples // batch_size

        x = tf.placeholder(tf.float32,[None,784],name = 'input')
        y = tf.placeholder(tf.float32,[None,10],name = 'y')
        keep_prob=tf.placeholder(tf.float32,name = 'keep_prob')
        lr = tf.Variable(self.learning_rate, dtype=tf.float32)
        is_training = tf.placeholder(tf.bool,name = 'is_training')

        if self.layer_num == 1:

            W1 = tf.Variable(tf.truncated_normal([784,hidden_cells],stddev=self.stddev))
            b1 = tf.Variable(tf.zeros([hidden_cells])+self.bias)
            matmul_1 = tf.matmul(x,W1)+b1
            if self.batch_norm == 1:
                matmul_1 = self.batch_norm_layer(matmul_1,is_training = is_training)
            if self.activation == 'tanh':
                L1 = tf.nn.tanh(matmul_1)
            elif self.activation == 'sigmoid':
                L1 = tf.nn.sigmoid(matmul_1)
            elif self.activation == 'leaky':
                L1 = tf.nn.leaky_relu(matmul_1)
            elif self.activation == 'relu':
                L1 = tf.nn.relu(matmul_1)
            else:
                L1 = tf.nn.swish(matmul_1)
            L1_drop = tf.nn.dropout(L1,keep_prob) 

            W2 = tf.Variable(tf.truncated_normal([hidden_cells,10],stddev=self.stddev))
            b2 = tf.Variable(tf.zeros([10])+self.bias)
            matmul_2 = tf.matmul(L1_drop,W2)+b2
            if self.batch_norm == 1:
                matmul_2 = self.batch_norm_layer(matmul_2,is_training = is_training)
            if self.activation == 'tanh':
                L2 = tf.nn.tanh(matmul_2)
            elif self.activation == 'sigmoid':
                L1 = tf.nn.sigmoid(matmul_2)
            elif self.activation == 'leaky':
                L1 = tf.nn.leaky_relu(matmul_2)
            elif self.activation == 'relu':
                L1 = tf.nn.relu(matmul_2)
            else:
                L1 = tf.nn.swish(matmul_2)
            L2_drop = tf.nn.dropout(L2,keep_prob)
            if self.loss == 'softmax':
                prediction = tf.nn.softmax(L2_drop,name = 'prediction')
            elif self.loss == 'softplus':
                prediction = tf.nn.softplus(L2_drop,name = 'prediction')
            elif self.loss == 'softsign':
                prediction = tf.nn.softsign(L2_drop,name = 'prediction')
            
        elif self.layer_num >= 2:
            W1 = tf.Variable(tf.truncated_normal([784,hidden_cells],stddev=self.stddev))
            b1 = tf.Variable(tf.zeros([hidden_cells])+self.bias)
            matmul_1 = tf.matmul(x,W1)+b1
            if self.batch_norm == 1:
                matmul_1 = self.batch_norm_layer(matmul_1,is_training = is_training)
            if self.activation == 'tanh':
                L1 = tf.nn.tanh(matmul_1)
            elif self.activation == 'sigmoid':
                L1 = tf.nn.sigmoid(matmul_1)
            elif self.activation == 'leaky':
                L1 = tf.nn.leaky_relu(matmul_1)
            elif self.activation == 'relu':
                L1 = tf.nn.relu(matmul_1)
            else:
                L1 = tf.nn.swish(matmul_1)
            L1_drop = tf.nn.dropout(L1,keep_prob) 
            Wn = []
            bn = []
            Ln = []
            Ln_drop = [L1_drop]

            for n in range(self.layer_num-1):
                 
                Wn.append(tf.Variable(tf.truncated_normal([hidden_cells,hidden_cells],stddev=self.stddev)))
                bn.append(tf.Variable(tf.zeros([hidden_cells])+self.bias))
                matmul_2 = tf.matmul(Ln_drop[n],Wn[n])+bn[n]
                if self.batch_norm == 1:
                    matmul_2 = self.batch_norm_layer(matmul_2,is_training = is_training)
                if self.activation == 'tanh':
                    Ln.append (tf.nn.tanh(matmul_2))
                elif self.activation == 'sigmoid':
                    Ln.append (tf.nn.sigmoid(matmul_2))
                elif self.activation == 'leaky':
                    Ln.append (tf.nn.leaky_relu(matmul_2))
                elif self.activation == 'relu':
                    Ln.append (tf.nn.relu(matmul_2))
                else:
                    Ln.append (tf.nn.swish(matmul_2))
                Ln_drop.append(tf.nn.dropout(Ln[n],keep_prob))         
            W2 = tf.Variable(tf.truncated_normal([hidden_cells,10],stddev=self.stddev))
            b2 = tf.Variable(tf.zeros([10])+self.bias)
            if self.loss == 'softmax':
                prediction = tf.nn.softmax(tf.matmul(Ln_drop[-1],W2)+b2,name = 'prediction')
            elif self.loss == 'softplus':
                prediction = tf.nn.softplus(tf.matmul(Ln_drop[-1],W2)+b2,name = 'prediction')
            elif self.loss == 'softsign':
                prediction = tf.nn.softsign(tf.matmul(Ln_drop[-1],W2)+b2,name = 'prediction')
       
        tf.add_to_collection(tf.GraphKeys.WEIGHTS,W1)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS,W2)
        if self.layer_num >= 2:
            if Wn:
                for W in Wn:
                    tf.add_to_collection(tf.GraphKeys.WEIGHTS,W)
               
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
        if self.nomorlization == 1:
            regularizer = tf.contrib.layers.l1_regularizer(scale=100/50000)
            reg_term = tf.contrib.layers.apply_regularization(regularizer)
            loss = loss + reg_term
        elif self.nomorlization == 2:
            regularizer = tf.contrib.layers.l2_regularizer(scale = 100/50000)
            reg_term = tf.contrib.layers.apply_regularization(regularizer)
            loss = loss + reg_term

        if self.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(lr)
        elif self.optimizer == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(lr)
        elif self.optimizer == 'Adadelta':
            optimizer = tf.train.AdadeltaOptimizer(lr)
        elif self.optimizer == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(lr)
        else:
            optimizer = tf.train.RMSPropOptimizer(lr)

        grads, variables = zip(*optimizer.compute_gradients(loss))
        if self.clip == 0:
            train_step = optimizer.minimize(loss)
        elif self.clip == 1:         
            grads, global_norm = tf.clip_by_global_norm(grads, 5)
            train_step = optimizer.apply_gradients(zip(grads, variables))
        elif self.clip == 2: 
            grads, global_norm = tf.clip_by_global_norm(grads, 10)
            train_step = optimizer.apply_gradients(zip(grads, variables))

        init = tf.global_variables_initializer()

        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        saver = tf.train.Saver(var_list = tf.global_variables())
        self.sess.run(init)
        steps = list()
        train_accuracy = list()
        test_accuracy = list()
        learning_rates = list()
        loss_list = list()
        step = 0
        for epoch in range(self.epoch):
            step_2 = 0
            self.sess.run(tf.assign(lr, self.learning_rate * (0.95 ** epoch)))
            for batch in range(1,n_batch):
                step += 1
                step_2 += 1
                batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
                loss_,_ = self.sess.run([loss,train_step],feed_dict={x:batch_xs,y:batch_ys,keep_prob:self.keep_prob,is_training:True}) 
                if batch % 10 == 0:
                    train_acc = self.sess.run(accuracy,feed_dict = {x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0,is_training:False})            
                    test_acc = self.sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0,is_training:False})
                    learning_rate = self.sess.run(lr)
                    steps.append(step)
                    train_accuracy.append(train_acc)
                    test_accuracy.append(test_acc)
                    learning_rates.append(learning_rate)
                    loss_list.append(loss_)
                    log_info =  "周期数：{}\n步数：{} / {}\n训练集准确率：{}\n测试集准确率：{}\n学习率：{}\n损失值：{}".format(epoch + 1,step_2,n_batch,train_acc,test_acc,learning_rate,loss_) 
                    self.breakSignal.emit(log_info)
                
            saver.save(self.sess,'mnist/mnist.ckpt')            
            
        finish_info = "训练完成!\n训练集准确率：{}\n测试集准确率：{}".format(train_acc,test_acc) 
        self.breakSignal.emit(finish_info)
        self.stopSignal.emit()
        fig = plt.figure(figsize = (10,10))
        ax1 = fig.add_subplot(4,1,1)
        plt.ylim(0.2,1)
        plt.xlim(0,steps[-1])
        plt.ylabel('train accuracy')
        plt.plot(steps,train_accuracy)
        ax2 = fig.add_subplot(4,1,2)
        plt.ylim(0.2,1)
        plt.xlim(0,steps[-1])
        plt.ylabel('test accuracy')
        plt.plot(steps,test_accuracy)
        ax3 = fig.add_subplot(4,1,3)
        plt.xlim(0,steps[-1])        
        plt.ylim(0.0006,0.001)
        plt.ylabel('learning rate')
        plt.plot(steps,learning_rates)
        ax4 = fig.add_subplot(4,1,4)
        plt.xlim(0,steps[-1])
        plt.ylim(0,5)
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.plot(steps,loss_list)
        plt.savefig('myfig.png')

    def close_thread(self):
        try:
            self.sess.close()
        except:
            pass
        


            

          #  self.sleep(1)
