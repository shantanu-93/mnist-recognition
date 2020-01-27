""" CNN
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
class_size = 10
batch_size = 128
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

X = tf.placeholder(tf.float32, [batch_size, 784], name='image') 
Y = tf.placeholder(tf.int32, name='label')
##### CNN #####
keeprate = 0.9
weights = {'w_c1' : tf.Variable(tf.random_normal([5,5,1,32])),
                     'w_c2' : tf.Variable(tf.random_normal([5,5,32,64])),
                     'w_fc' : tf.Variable(tf.random_normal([7*7*64,1024])),
                     'w_out'     : tf.Variable(tf.random_normal([1024, class_size]))}

bias = {'b_c1' : tf.Variable(tf.random_normal([32])),
                'b_c2' : tf.Variable(tf.random_normal([64])),
                'b_fc' : tf.Variable(tf.random_normal([1024])),
                'b_out'    : tf.Variable(tf.random_normal([class_size]))}

x_reshaped = tf.reshape(X, shape=[-1,28,28,1])

convolove1 = tf.nn.relu(tf.nn.conv2d(x_reshaped, weights['w_c1'], strides=[1,1,1,1], padding='SAME') + bias['b_c1'])
convolove1 = tf.nn.max_pool(convolove1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

convolve2 = tf.nn.relu(tf.nn.conv2d(convolove1, weights['w_c2'], strides=[1,1,1,1], padding='SAME') + bias['b_c2'])
convolve2 = tf.nn.max_pool(convolve2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

fully_connected = tf.reshape(convolve2,[-1, 7*7*64])
fully_connected = tf.nn.relu(tf.matmul(fully_connected, weights['w_fc'])+bias['b_fc'])
fully_connected = tf.nn.dropout(fully_connected, keeprate)

final_output = tf.matmul(fully_connected, weights['w_out'])+bias['b_out']

n_epochs = 15
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=Y, name='loss')
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer().minimize(loss)

preds = tf.nn.softmax(final_output)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/cnn', tf.get_default_graph())
with tf.Session() as sess:
        start_time = time.time()
        sess.run(tf.global_variables_initializer())        
        train_batches = int(mnist.train.num_examples/batch_size)
        
        for i in range(n_epochs): 
                total_loss = 0

                for j in range(train_batches):    
                        X_batch, Y_batch = mnist.train.next_batch(batch_size)
                        _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch}) 
                        total_loss += loss_batch
                print('Average loss epoch {0}: {1}'.format(i, total_loss/train_batches))
        print('Total time: {0} seconds'.format(time.time() - start_time))

        total_correct_preds = 0

        test_batches = int(mnist.test.num_examples/batch_size)
        for i in range(test_batches):
                X_batch, Y_batch = mnist.test.next_batch(batch_size)
                accuracy_batch = sess.run(accuracy, {X: X_batch, Y:Y_batch})
                total_correct_preds += accuracy_batch

        print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))


writer.close()