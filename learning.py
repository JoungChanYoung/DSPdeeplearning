import tensorflow as tf
import os
import numpy as np

tf.set_random_seed(777)

# parameter
num_classes = 5  # 분류의 개수
learning_rate = 0.01
hidden_size = 20  # 출력의 개수(0또는 1)
sequence_length = 20  # 한 음원의 trimmed 된 개수
input_dim = 13  # input값의 개수 mfcc 13개의 벡터
output_dim = 5  # FC통과후의 output개수


idx2scene = ['1.car', '2.train', '3.gun', '4.rainy', '5.windy']

# data
x = np.loadtxt('car_mfcc.csv', delimiter=',')
x2 = np.loadtxt('train_mfcc.csv',delimiter=',')
x3 = np.loadtxt('gun_mfcc.csv', delimiter=',')
x4 = np.loadtxt('rainy_mfcc.csv', delimiter=',')
x5 = np.loadtxt('windy_mfcc.csv', delimiter=',')

# data정리

x_train=np.empty([0,13])#append를위해 empty배열 생성
x_test=np.empty([0,13])#append를 위해
x_valid=np.empty([0,13])#append를 위해

y_train=[]
y_test=[]
y_valid = []

rate = 40
num = sequence_length * rate
valid_num = 100

x_valid = np.append(x_train,x[:valid_num,:],axis=0)
x_train=np.append(x_train,x[valid_num:(len(x)-num),:],axis=0)
x_test=np.append(x_test,x[(len(x)-num):len(x),:],axis=0)

x_valid = np.append(x_train,x2[:valid_num,:],axis=0)
x_train=np.append(x_train,x2[valid_num:(len(x2)-num),:],axis=0)
x_test=np.append(x_test,x2[(len(x2)-num):len(x2),:],axis=0)

x_valid = np.append(x_train,x3[:valid_num,:],axis=0)
x_train=np.append(x_train,x3[valid_num:(len(x3)-num),:],axis=0)
x_test=np.append(x_test,x3[(len(x3)-num):len(x3),:],axis=0)


x_valid = np.append(x_train,x4[:valid_num,:],axis=0)
x_train=np.append(x_train,x4[valid_num:(len(x4)-num),:],axis=0)
x_test=np.append(x_test,x4[(len(x4)-num):len(x4),:],axis=0)
'''
x_valid = np.append(x_train,x5[:valid_num,:],axis=0)
x_train=np.append(x_train,x5[valid_num:(len(x5)-num),:],axis=0)
x_test=np.append(x_test,x5[(len(x5)-num):len(x5),:],axis=0)
'''
x_train=np.reshape(x_train,[-1,20,13])#None, seq_len,input_dim
x_test=np.reshape(x_test,[-1,20,13])
x_valid=np.reshape(x_test,[-1,20,13])


y_valid = np.append(y_train,np.full([int(valid_num/sequence_length),1],0))
y_valid = np.append(y_train,np.full([int(valid_num/sequence_length),1],1))
y_valid = np.append(y_train,np.full([int(valid_num/sequence_length),1],2))
y_valid = np.append(y_train,np.full([int(valid_num/sequence_length),1],3))
'''
y_valid = np.append(y_train,np.full([int(valid_num/sequence_length),1],4))
'''                    
y_train=np.append(y_train,np.full([int(len(x)/sequence_length)-rate-int(valid_num/sequence_length),1],0))
y_test=np.append(y_test,np.full([rate,1],0))                    
y_train=np.append(y_train,np.full([int(len(x2)/sequence_length)-rate-int(valid_num/sequence_length),1],1))
y_test=np.append(y_test,np.full([rate,1],1))
y_train=np.append(y_train,np.full([int(len(x3)/sequence_length)-rate-int(valid_num/sequence_length),1],2))
y_test=np.append(y_test,np.full([rate,1],2))
y_train=np.append(y_train,np.full([int(len(x4)/sequence_length)-rate-int(valid_num/sequence_length),1],3))
y_test=np.append(y_test,np.full([rate,1],3))
'''
y_train=np.append(y_train,np.full([int(len(x5)/sequence_length)-rate-int(valid_num/sequence_length),1],4))
y_test=np.append(y_test,np.full([rate,1],4))
'''
y_train=np.reshape(y_train,[-1,1])
y_test=np.reshape(y_test,[-1,1])
y_valid=np.reshape(y_test,[-1,1])

# learning

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])
Y = tf.placeholder(tf.int64, [None, 1])
Y_one_hot = tf.one_hot(Y, num_classes)
_Y_one_hot = tf.reshape(Y_one_hot, [-1, num_classes])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True
                                    , activation=None)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)

Y_pred_softmax = tf.nn.softmax(Y_pred)

loss = tf.reduce_mean(-tf.reduce_sum(_Y_one_hot * tf.log(Y_pred_softmax), axis=1))

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

_Y = tf.reshape(Y, [-1])
prediction = tf.argmax(Y_pred_softmax, axis=1)
correct_prediction = tf.equal(prediction, _Y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(100):
    _, _loss, acc = sess.run([train, loss, accuracy], feed_dict={X: x_train, Y: y_train})
    if (i % 10 == 0):
        print("Step: {:5}\t Loss: {:.7f}\tAcc: {:.2%}".format(i, _loss, acc))
    if (i % 20 == 0):
        temp = 0
        pred = sess.run(prediction, feed_dict={X: x_valid})
        for p, y in zip(pred, y_valid.flatten()):
            print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
            temp += int(p == int(y))
        print("Accuracy :{}".format(temp / len(y_valid)))

pred = sess.run(prediction, feed_dict={X: x_test})

temp = 0
for p, y in zip(pred, y_test.flatten()):
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
    temp += int(p == int(y))

print("Accuracy :{}".format(temp / len(y_test)))

# testPredict = sess.run(sess.run(Y_pred, feed_dict={X:x_train}))
        
print("finish")


