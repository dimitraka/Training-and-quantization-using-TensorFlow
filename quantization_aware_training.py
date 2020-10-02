import tempfile
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
from keras.utils import to_categorical


# load train and test dataset
def load_dataset():
	# This is a dataset of 50,000 32x32 color training images
	# and 10,000 test images, labeled over 10 categories.
	(trainX, trainY), (testX, testY) = cifar10.load_data()

	# keep the labels of the testing set unchanged as well
	testY_unchanged = testY
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY, testY_unchanged


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


def clone_pretrained_model(model):

	import tensorflow_model_optimization as tfmot

	quantize_model = tfmot.quantization.keras.quantize_model

	# q_aware stands for for quantization aware.
	q_aware_model = quantize_model(model)

	# `quantize_model` requires a recompile.
	q_aware_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	q_aware_model.summary()

	return q_aware_model


def train_q_aware_model(q_aware_model, train_images, train_labels, test_images, test_labels):
	train_images_subset = train_images[0:50000] # out of 50000
	train_labels_subset = train_labels[0:50000]

	q_aware_model.fit(train_images_subset, train_labels_subset,
	                  batch_size=500, epochs=10, validation_split=0.1)

	_, baseline_model_accuracy = model.evaluate(
	    test_images, test_labels, verbose=0)

	_, q_aware_model_accuracy = q_aware_model.evaluate(
	   test_images, test_labels, verbose=0)

	print('Baseline test accuracy:', baseline_model_accuracy)
	print('Quant test accuracy:', q_aware_model_accuracy)

	return q_aware_model, q_aware_model_accuracy, baseline_model_accuracy


def create_quantized_model(q_aware_model):
	converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]

	quantized_tflite_model = converter.convert()

	return quantized_tflite_model


def evaluate_model(interpreter, test_images, test_labels):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on every image in the "test" dataset.
  prediction_images = []
  for i, test_image in enumerate(test_images):
    if i % 1000 == 0:
      print('Evaluated on {n} results so far.'.format(n=i))
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the predicted value with highest
    # probability.
    output = interpreter.tensor(output_index)
    predicted_value = np.argmax(output()[0])
    prediction_images.append(predicted_value)

  print('\n')
  # Compare prediction results with ground truth labels to calculate accuracy.
  prediction_images = np.array(prediction_images)
  #prediction_images = to_categorical(prediction_images)
  accurate_count = 0
  for index in range(len(prediction_images)):
    if prediction_images[index] == test_labels[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(prediction_images)
  return accuracy


def print_weights_data_type(model):
	no_trainable_param = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
	print(no_trainable_param)
	var = [v for v in model.trainable_variables][0]
	print(var.dtype)


def layer_details_tflite(quantized_tflite_model):

	# Load TFLite model and allocate tensors.
	interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
	interpreter.allocate_tensors()

	# Get input and output tensors.
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()


	# get details for each layer
	all_layers_details = interpreter.get_tensor_details()
	#print(all_layers_details)

	for layer in all_layers_details:

	     # to store layer's metadata in group's metadata
	     print("\nname:",layer['name'])
	     print("shape:", layer['shape'])
	     print("dtype", layer['dtype'])
	     print("quantization", layer['quantization'])


#See 4x smaller model from quantization
def size_reduction(model):

	# Create float TFLite model.
	float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
	float_tflite_model = float_converter.convert()

	with open('models/float_model.tflite', 'wb') as f:
	  f.write(float_tflite_model)

	float_TFLite_file = 'models/float_model.tflite'
	quant_TFLite_file = 'models/qat_model.tflite'
	float_file = 'models/mobilenet.h5'

	print("TFLite float model in Mb:", os.path.getsize(float_TFLite_file) / float(2**20))
	print("Quantized TFLite model in Mb:", os.path.getsize(quant_TFLite_file) / float(2**20))
	print("Float model in Mb:", os.path.getsize(float_file) / float(2**20))



# load dataset, load pre-trained model, quantize model and evaluate
def quantization_aware_training():

	# load dataset
	trainX, trainY, testX, testY, testY_unchanged = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)

	# Load pretrained model
	model = tf.keras.models.load_model('models/mobilenet.h5')
	# Re-evaluate the model that was just loaded
	loss,acc = model.evaluate(testX, testY, verbose=1)
	print("Restored model, accuracy: {:5.2f}%".format(100*acc))
	#print_weights_data_type(model)

	q_aware_model = clone_pretrained_model(model)

	q_aware_model, q_aware_model_accuracy, baseline_model_accuracy = train_q_aware_model(q_aware_model, trainX, trainY, testX, testY)
	#print_weights_data_type(q_aware_model)

	quantized_tflite_model = create_quantized_model(q_aware_model)
	#layer_details_tflite(quantized_tflite_model)

	# Save the TFLite
	# qat stands for quantization-aware training
	with open('models/qat_model.tflite', 'wb') as f:
	  f.write(quantized_tflite_model)

	# Run the TFlite
	#interpreter = tf.lite.Interpreter(model_path="/models/qat_model.tflite") # if you want to load the model
	interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
	interpreter.allocate_tensors()

	test_accuracy = evaluate_model(interpreter, testX, testY_unchanged)

	print('Quant TFLite test_accuracy:', test_accuracy)
	print('Quant TF test accuracy:', q_aware_model_accuracy)
	print('TF test accuracy:', baseline_model_accuracy)

	# See size of each model
	size_reduction(model)


# entry point
# load pre-trained model and quantize
quantization_aware_training()
