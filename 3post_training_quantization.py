import tempfile
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
from keras.utils import to_categorical


# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = cifar10.load_data()

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


def representative_data_gen():
	for input_value in tf.data.Dataset.from_tensor_slices(trainX).batch(1).take(100):
		yield [input_value]


def convert_to_TFLite(model, trainX):

	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.representative_dataset = representative_data_gen
	# Ensure that if any ops can't be quantized, the converter throws an error
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	# Set the input and output tensors to uint8 (APIs added in r2.3)
	converter.inference_input_type = tf.uint8
	converter.inference_output_type = tf.uint8

	tflite_model_quant = converter.convert()

	# The internal quantization remains the same as above, but you can see the input and output tensors are now integer format
	interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
	input_type = interpreter.get_input_details()[0]['dtype']
	print('input: ', input_type)
	output_type = interpreter.get_output_details()[0]['dtype']
	print('output: ', output_type)

	return tflite_model_quant


# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices, test_images, test_labels):

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_image_indices),), dtype=int)
  for i, test_image_index in enumerate(test_image_indices):
    test_image = test_images[test_image_index]
    test_label = test_labels[test_image_index]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      test_image = test_image / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    predictions[i] = output.argmax()

  return predictions



def evaluate_model(tflite_file, model_type, test_images, test_labels):

  test_image_indices = range(test_images.shape[0])
  predictions = run_tflite_model(tflite_file, test_image_indices, test_images, test_labels)

  #accuracy = (np.sum(test_labels== predictions) * 100) / len(test_images)
  accurate_count = 0
  for index in range(len(predictions)):
    if predictions[index] == test_labels[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(predictions)

  print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (
      model_type, accuracy, len(test_images)))


#---------------------entry point---------------------------------

# load dataset
trainX, trainY, testX, testY, testY_unchanged = load_dataset()
# prepare pixel data
trainX, testX = prep_pixels(trainX, testX)

# Load pretrained model
model = tf.keras.models.load_model('mobilenet.h5')
# Re-evaluate the model that was just loaded
loss,acc = model.evaluate(testX, testY, verbose=1)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# Convert to TFLite using integer only quantization
tflite_model_quant = convert_to_TFLite(model, trainX)

# Save model
with open('post_trainining_quantized_model.tflite', 'wb') as f:
  f.write(tflite_model_quant)

tflite_model_quant_file = 'post_trainining_quantized_model.tflite'
tflite_model_file = 'model.tflite'

# Run the TFlite
evaluate_model(tflite_model_quant_file, "Quantized TFLite", testX, testY_unchanged )
evaluate_model(tflite_model_file, "Not Quantized TFLite", testX, testY_unchanged )

print("Quantized TFLite model in Mb:", os.path.getsize(tflite_model_quant_file) / float(2**20))
print("Not quantized TFLite model in Mb:", os.path.getsize(tflite_model_file) / float(2**20))
