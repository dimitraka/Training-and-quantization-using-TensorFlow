# baseline model with dropout on the cifar10 dataset
import sys
import os
import tensorflow as tf
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.layers import Dense

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY


# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm


# define cnn model
def define_model():
	model = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights=None)

	# Uncomment if there are incompatibilities with the model's shapes
	'''f_flat = tf.keras.layers.GlobalAveragePooling2D()(model.output)
	fc = Dense(units=2048,activation="relu")(f_flat)
	logit = Dense(units=10, activation="softmax")(fc)
	model = tf.keras.Model(model.inputs,logit)'''

	# compile model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model


# load dataset, load, train and save model
def train_model():

	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)

	# load or define model
	if (os.path.isfile("models/mobilenet.h5") == False):
		print("There is no model saved in this folder. A new model will be defined and trained.\n")
		model = define_model()
	else:
		print("Found a model saved in this folder. This model will be loaded and trained.\n")
		# load pretrained model
		model = tf.keras.models.load_model('models/mobilenet.h5')
		# Re-evaluate the model
		loss,acc = model.evaluate(testX, testY, verbose=1)
		print("Restored model, accuracy: {:5.2f}%".format(100*acc))


	# fit model
	history = model.fit(trainX, trainY, epochs=10, verbose=2)
	# evaluate model
	_, acc = model.evaluate(testX, testY, verbose=2)

	#save model
	model.save('models/mobilenet.h5')
	#check accuracy of saved model
	model = tf.keras.models.load_model('models/mobilenet.h5')
	loss,acc = model.evaluate(testX, testY, verbose=1)
	print("Saved model, accuracy: {:5.2f}%".format(100*acc))


# entry point
# train and save the model
train_model()
