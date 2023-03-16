import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam
from keras.datasets import mnist

def loadData():
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	size = 10000
	x_train = x_train[0:size]
	y_train = y_train[0:size]
	x_train = x_train.reshape(size,28 * 28)
	x_test = x_test.reshape(x_test.shape[0],28 * 28)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	y_train = np_utils.to_categorical(y_train,10)
	y_test = np_utils.to_categorical(y_test,10)
	x_train = x_train / 255
	x_test = x_test / 255

	return (x_train, y_train),(x_test,y_test)

def run():
	(x_train, y_train),(x_test,y_test) = loadData()

	model = Sequential()
	units = 28 * 28

	model.add(Dense(input_dim = units, units = units, activation = "relu"))
	for i in range(2):
		model.add(Dense(units = units, activation = 'relu'))

	model.add(Dense(units = 10, activation = 'softmax'))

	model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])

	model.fit(x_train,y_train,batch_size = 100, epochs = 20)

	result = model.evaluate(x_train,y_train,batch_size = 100)
	print('\nTrain ACC: %.2f%%' %(result[1] * 100))
	result = model.evaluate(x_test,y_test,batch_size = 100)
	print('\nTest Acc: %.2f%%' %(result[1] * 100))

if __name__ == '__main__':
	run()
