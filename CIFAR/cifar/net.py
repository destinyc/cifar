import tensorflow as tf
from config import Config
from read_data import Read_cifar10
import time

class VGG_net:
    def __init__(self):
        self.lr = Config.lr
        self.batch_size = Config.batch_size
        self.iteration = Config.iteration
        self.epoch = Config.train_epoch
        self.weight_decay = Config.weight_decay
        self.momentum_rate = Config.momentum_rate

        read_data = Read_cifar10()
        self.train_data, self.train_label, self.test_data, self.test_label = read_data.read_data()

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool(self, input, k_size=1, stride=1, name=None):
        return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                              padding='SAME', name=name)

    def batch_norm(self, input):
        return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                            is_training=self.training_flag, updates_collections=None)

    def learning_rate_schedule(self, epoch_num):
        if epoch_num < 81:
            return 0.1
        elif epoch_num < 121:
            return 0.01
        else:
            return 0.001

    def generate_batch_train_data(self, batch_index):

        return self.train_data[batch_index * self.batch_size : (batch_index + 1) * self.batch_size], \
               self.train_label[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]

    def run_testing(self, sess):

        acc = sess.run(self.accuracy, feed_dict={self.input: self.test_data[:1000],
                                                  self.label: self.test_label[:1000],
                                                  self.keep_prob: 1.0, self.training_flag: False})

        print('test accuracy : %f'%(acc))

    def build_vgg_16(self):
        self.input = tf.placeholder(dtype = tf.float32, shape = [None, 32, 32, 3])
        self.label = tf.placeholder(dtype = tf.float32, shape = [None, 10])
        self.learning_rate = tf.placeholder(dtype = tf.float32)
        self.keep_prob = tf.placeholder(dtype = tf.float32)
        self.training_flag = tf.placeholder(dtype = tf.bool)

        #build network

        W_conv1_1 = tf.get_variable('conv1_1', shape=[3, 3, 3, 64],
                                    initializer=tf.contrib.keras.initializers.he_normal())
        b_conv1_1 = self.bias_variable([64])
        output = tf.nn.relu(self.batch_norm(self.conv2d(self.input, W_conv1_1) + b_conv1_1))

        W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, 64, 64],
                                    initializer=tf.contrib.keras.initializers.he_normal())
        b_conv1_2 = self.bias_variable([64])
        output = tf.nn.relu(self.batch_norm(self.conv2d(output, W_conv1_2) + b_conv1_2))
        output = self.max_pool(output, 2, 2, "pool1")

        W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, 64, 128],
                                    initializer=tf.contrib.keras.initializers.he_normal())
        b_conv2_1 = self.bias_variable([128])
        output = tf.nn.relu(self.batch_norm(self.conv2d(output, W_conv2_1) + b_conv2_1))

        W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, 128, 128],
                                    initializer=tf.contrib.keras.initializers.he_normal())
        b_conv2_2 = self.bias_variable([128])
        output = tf.nn.relu(self.batch_norm(self.conv2d(output, W_conv2_2) + b_conv2_2))
        output = self.max_pool(output, 2, 2, "pool2")

        W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 128, 256],
                                    initializer=tf.contrib.keras.initializers.he_normal())
        b_conv3_1 = self.bias_variable([256])
        output = tf.nn.relu(self.batch_norm(self.conv2d(output, W_conv3_1) + b_conv3_1))

        W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 256, 256],
                                    initializer=tf.contrib.keras.initializers.he_normal())
        b_conv3_2 = self.bias_variable([256])
        output = tf.nn.relu(self.batch_norm(self.conv2d(output, W_conv3_2) + b_conv3_2))

        W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3, 256, 256],
                                    initializer=tf.contrib.keras.initializers.he_normal())
        b_conv3_3 = self.bias_variable([256])
        output = tf.nn.relu(self.batch_norm(self.conv2d(output, W_conv3_3) + b_conv3_3))

        W_conv3_4 = tf.get_variable('conv3_4', shape=[3, 3, 256, 256],
                                    initializer=tf.contrib.keras.initializers.he_normal())
        b_conv3_4 = self.bias_variable([256])
        output = tf.nn.relu(self.batch_norm(self.conv2d(output, W_conv3_4) + b_conv3_4))
        output = self.max_pool(output, 2, 2, "pool3")

        W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 256, 512],
                                    initializer=tf.contrib.keras.initializers.he_normal())
        b_conv4_1 = self.bias_variable([512])
        output = tf.nn.relu(self.batch_norm(self.conv2d(output, W_conv4_1) + b_conv4_1))

        W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3, 512, 512],
                                    initializer=tf.contrib.keras.initializers.he_normal())
        b_conv4_2 = self.bias_variable([512])
        output = tf.nn.relu(self.batch_norm(self.conv2d(output, W_conv4_2) + b_conv4_2))

        W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, 512, 512],
                                    initializer=tf.contrib.keras.initializers.he_normal())
        b_conv4_3 = self.bias_variable([512])
        output = tf.nn.relu(self.batch_norm(self.conv2d(output, W_conv4_3) + b_conv4_3))

        W_conv4_4 = tf.get_variable('conv4_4', shape=[3, 3, 512, 512],
                                    initializer=tf.contrib.keras.initializers.he_normal())
        b_conv4_4 = self.bias_variable([512])
        output = tf.nn.relu(self.batch_norm(self.conv2d(output, W_conv4_4)) + b_conv4_4)
        output = self.max_pool(output, 2, 2)

        W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 512, 512],
                                    initializer=tf.contrib.keras.initializers.he_normal())
        b_conv5_1 = self.bias_variable([512])
        output = tf.nn.relu(self.batch_norm(self.conv2d(output, W_conv5_1) + b_conv5_1))

        W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, 512, 512],
                                    initializer=tf.contrib.keras.initializers.he_normal())
        b_conv5_2 = self.bias_variable([512])
        output = tf.nn.relu(self.batch_norm(self.conv2d(output, W_conv5_2) + b_conv5_2))

        W_conv5_3 = tf.get_variable('conv5_3', shape=[3, 3, 512, 512],
                                    initializer=tf.contrib.keras.initializers.he_normal())
        b_conv5_3 = self.bias_variable([512])
        output = tf.nn.relu(self.batch_norm(self.conv2d(output, W_conv5_3) + b_conv5_3))

        W_conv5_4 = tf.get_variable('conv5_4', shape=[3, 3, 512, 512],
                                    initializer=tf.contrib.keras.initializers.he_normal())
        b_conv5_4 = self.bias_variable([512])
        output = tf.nn.relu(self.batch_norm(self.conv2d(output, W_conv5_4) + b_conv5_4))

        # output = tf.contrib.layers.flatten(output)
        output = tf.reshape(output, [-1, 2 * 2 * 512])

        W_fc1 = tf.get_variable('fc1', shape=[2048, 4096], initializer=tf.contrib.keras.initializers.he_normal())
        b_fc1 = self.bias_variable([4096])
        output = tf.nn.relu(self.batch_norm(tf.matmul(output, W_fc1) + b_fc1))
        output = tf.nn.dropout(output, self.keep_prob)

        W_fc2 = tf.get_variable('fc7', shape=[4096, 4096], initializer=tf.contrib.keras.initializers.he_normal())
        b_fc2 = self.bias_variable([4096])
        output = tf.nn.relu(self.batch_norm(tf.matmul(output, W_fc2) + b_fc2))
        output = tf.nn.dropout(output, self.keep_prob)

        W_fc3 = tf.get_variable('fc3', shape=[4096, 10], initializer=tf.contrib.keras.initializers.he_normal())
        b_fc3 = self.bias_variable([10])
        output = tf.nn.relu(self.batch_norm(tf.matmul(output, W_fc3) + b_fc3))

        #loss and accuracy
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = self.label, logits = output)

        self.loss = tf.reduce_mean(cross_entropy) + self.weight_decay* tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        self.train_step = tf.train.MomentumOptimizer(self.learning_rate, self.momentum_rate, use_nesterov=True). \
            minimize(self.loss)

        # self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(1, self.epoch + 1):

                lr = self.learning_rate_schedule(epoch)
                for iteration in range(1, self.iteration + 1):
                    train_data, train_label = self.generate_batch_train_data(iteration)

                    _, loss = sess.run([self.train_step, self.loss], feed_dict = {
                        self.input: train_data,
                        self.label: train_label,
                        self.keep_prob: Config.keep_prob, self.training_flag: True,
                        self.learning_rate : lr
                    })
                    print('iteration : %d, loss : %f'%(iteration, loss))

                    if iteration % 100 == 0:
                        self.run_testing(sess)

           
                saver.save(sess, save_path='../model/vgg16_%d.ckpt' % (epoch))













class Le_net:
    def __init__(self):
        self.lr = Config.lr
        self.batch_size = Config.batch_size
        self.iteration = Config.iteration
        self.epoch = Config.train_epoch
        self.weight_decay = Config.weight_decay

        read_data = Read_cifar10()
        self.train_data, self.train_label, self.test_data, self.test_label = read_data.read_data()

    def generate_batch_train_data(self, batch_index):

        return self.train_data[batch_index * self.batch_size : (batch_index + 1) * self.batch_size], \
               self.train_label[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]


    def build_Le_net(self):
        self.input = tf.placeholder(dtype = tf.float32, shape = [None, 32, 32, 3])
        self.label = tf.placeholder(dtype = tf.float32, shape = [None, 10])
        self.keep_prob = tf.placeholder(dtype = tf.float32)
        #第一个卷积层
        W1 = tf.get_variable(name = 'conv1_W', shape = [3,3,3,32], dtype = tf.float32,
                            initializer = tf.contrib.layers.xavier_initializer_conv2d())
        b1 = tf.get_variable(name = 'conv1_b', shape = [32], dtype = tf.float32,
                             initializer=tf.constant_initializer(0.0))

        conv1_output = tf.nn.bias_add(tf.nn.conv2d(self.input, W1, [1,1,1,1], padding = 'SAME'), b1)
        conv1_output = tf.nn.relu(conv1_output)
        pool1_output = tf.nn.max_pool(conv1_output, [1,2,2,1] ,[1,2,2,1], padding = 'SAME')

        #第二个卷积层
        W2 = tf.get_variable(name='conv2_W', shape=[3, 3, 32, 64], dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b2 = tf.get_variable(name='conv2_b', shape=[64], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        conv2_output = tf.nn.bias_add(tf.nn.conv2d(pool1_output, W2, [1, 1, 1, 1], padding='SAME'), b2)
        conv2_output = tf.nn.relu(conv2_output)
        pool2_output = tf.nn.max_pool(conv2_output, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        #第三个卷积层
        W3 = tf.get_variable(name='conv3_W', shape=[3, 3, 64, 128], dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b3 = tf.get_variable(name='conv3_b', shape=[128], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        conv3_output = tf.nn.bias_add(tf.nn.conv2d(pool2_output, W3, [1, 1, 1, 1], padding='SAME'), b3)
        conv3_output = tf.nn.relu(conv3_output)
        pool3_output = tf.nn.max_pool(conv3_output, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        shape = pool3_output.get_shape()
        flatten_shape = shape[1].value * shape[2].value * shape[3].value
        print('...',flatten_shape)
        fc_input = tf.reshape(pool3_output, [-1, flatten_shape])

        #第一个全连接层
        fc_w1 = tf.get_variable(name = 'fc1_w', shape = [fc_input.shape[-1], 128], dtype = tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
        fc_b1 = tf.get_variable(name = 'fc1_b', shape = [128], dtype = tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())

        fc1_output = tf.nn.relu(tf.matmul(fc_input, fc_w1) + fc_b1)
        fc1_output = tf.nn.dropout(fc1_output, keep_prob = self.keep_prob)

        #第二个全连接层
        fc_w2 = tf.get_variable(name='fc2_w', shape=[128, 10], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
        fc_b2 = tf.get_variable(name='fc2_b', shape=[10], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())

        output = tf.nn.relu(tf.matmul(fc1_output, fc_w2) + fc_b2)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = self.label, logits = output)
        self.loss = tf.reduce_mean(cross_entropy) + self.weight_decay * \
                            tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(1, self.epoch + 1):

                for iteration in range(self.iteration):

                    batch_data, batch_label = self.generate_batch_train_data(iteration)
                    _, loss = sess.run([self.train_step, self.loss], feed_dict={
                        self.input : batch_data, self.label : batch_label,
                        self.keep_prob : Config.keep_prob
                    })
                    print('iteration ; %d, loss : %f'%(iteration, loss))

                    if iteration % 100 == 0 :
                        acc = sess.run(self.accuracy, feed_dict={
                        self.input : self.test_data[:100], self.label : self.test_label[:100],
                        self.keep_prob : 1.0
                        })
                        print('test accuracy is :', acc)

                saver.save(sess, save_path='model\\vgg16_%d.ckpt' % (epoch))


















