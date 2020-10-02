## An end-to-end training and quantization example using TensorFlow ##

Code includes training of the model MobileNetV2 using the CIFAR10 dataset and full-integer quantization with 2 methods: post training quantization and quantization-aware training. 

### Installations ###

- Install TensorFlow (newest version): `pip install --upgrade tensorflow`

- Install Keras: `pip install keras`

### Files  ###

- Code files

  - <b>training.py</b>: Load dataset CIFAR10, train and evaluate the model MobileNetV2
  - <b>quantization_aware_training.py</b>: Quantize the pretrained model with quantization-aware training
  - <b>post_training_quantization.py</b>: Quantize the pretrained model with post-training quantization


- Folder <b> models </b> contains the following models:

  - <b>mobilenet.h5</b>: pretrained model mobilenet
  - <b>float_model.tflite</b>: TFLite model without quantization
  - <b>qat_model.tflite</b>: TFLite model after quantization-aware training
  - <b>post_training_model.tflite</b>: TFLite model after post training quantization

### Helper functions ###

- `model.summary()`: If you would like to see a summary of a model, i.e. the layers, the shapes, the trainable parameters etc.

- `print_weights_data_type(model)`: If you would like to see the data types of the weights of a `model` anytime during <b><i> quantization-aware training </i></b>

- `layer_details_tflite(model)`: If you would like to see info about the layers of a `model` i.e. name of the layers, shape, data type, quantization, anytime during <b><i> quantization-aware training </i></b>
