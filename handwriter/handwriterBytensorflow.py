import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SaveModelFolder = 'Model/Mnist'
#read data
mnist = input_data.read_data_sets('Mnist_data/',one_hot = True)

'''
	def addlayer(inputs, in_size, out_size, activation_function = None):
		W = tf.Variable(tf.truncated_normal([in_size,out_size],mean = 0, stddev = 0.2)
		b = tf.Variable(tf.zeros[1,out_size] + 0.1)

		Wx_plus_b = tf.matmul(inputs,W) + b

		if activation is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)
		return outputs
'''
##############################
#train 
##############################


tf.reset_default_graph()

#定义占位符
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([10]))

#定义分类器
pred = tf.nn.softmax(tf.matmul(x,W) + b)

#define loss function
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred),reduction_indices = 1))

learn_rate = 0.01
#define gradient 
optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

#define model save moduler
saver = tf.train.Saver()
model_path = SaveModelFolder

training_epochs = 20
batch_size = 10
display_step = 1

#execute
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for epoch in range(training_epochs):
		avg_cost = 0
		total_batch = int(mnist.train.num_examples/batch_size)

		for i in range(total_batch):
			batch_xs,batch_ys = mnist.train.next_batch(batch_size)
			_,c = sess.run([optimizer,cost],feed_dict = {x:batch_xs,y:batch_ys})
			avg_cost += c / total_batch

		if (epoch + 1) % display_step == 0:
			print("Epoch:" , epoch + 1, "cost = ",avg_cost)
			#构造bool型变量用于判断所有测试样本与其真是类别的匹配情况
			correct_prediction = tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
			#将bool型变量转换为float型并计算均值
			accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	result = accuracy.eval({x:mnist.test.images, y:mnist.test.labels})
	print(result)


	save_path = saver.save(sess,model_path)
	print("finished")

print("staring testing")

tf.reset_default_graph()
#定义占位符
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(x,W) + b)

#define loss function
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred),reduction_indices = 1))

learn_rate = 0.01

optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

#define model save moduler
saver1 = tf.train.Saver()
model_path = SaveModelFolder

with tf.Session() as sess2:
	sess2.run(tf.global_variables_initializer())

	saver1.restore(sess2,model_path)
	correct_prediction = tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	#这个写法等同于sess.run(accuract,feed_dict = {x:batch_xs,y:batch_ys})
	result = accuracy.eval({x:mnist.test.images,y:mnist.test.labels})
	print(result)

	output = tf.arg_max(pred,1)
	batch_xs,batch_ys = mnist.train.next_batch(2)
	outputval = sess2.run(output,feed_dict = {x:batch_xs, y:batch_ys})
	print(outputval,batch_ys)

	import pylab
	im = batch_xs[0]
	im = im.reshape(28,28)
	pylab.imshow(im)
	pylab.show()

	im = batch_xs[1]
	im = im.reshape(28,28)
	pylab.imshow(im)
	pylab.show()










