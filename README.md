# Happy-and-Sad-Image-Classifier
This Model predicts the input image is happy or sad.This project involves happy and sad image classification. The classification was conducted using Python programming language and its libraries, such as pandas, NumPy, and TensorFlow for building and training machine learning models, Keras for building and training deep learning models in TensorFlow. 

# Required Modules 
1. TensorFlow: TensorFlow is an open-source machine learning framework developed by Google that is widely used for building and training machine learning models, particularly neural networks.
2. Keras: This is a high-level API for building and training deep learning models in TensorFlow. Keras is an independent open-source deep learning library that was integrated into TensorFlow to provide a user-friendly and intuitive interface for creating neural networks. So, when you import tensorflow.keras, you get access to Keras functionality within TensorFlow.
3. NumPy
4. Matplotlib
5. OpenCV
6. Python OS module

# About Deep Learning Model Layers which is used in this project
1. Conv2D: This layer class is used to create convolutional layers in a neural network. Convolutional layers are fundamental for processing grid-like data, such as images. They apply filters to input data to detect features and patterns within the data.
2. MaxPooling2D: MaxPooling2D is a type of layer used in convolutional neural networks. It's typically applied after convolutional layers to reduce the spatial dimensions of the data and retain the most important information. Max pooling computes the maximum value within a small region of the input, effectively down-sampling the data.
3. Dense: The Dense layer represents a fully connected layer in a neural network. It's also known as a feedforward layer or a fully connected layer because each neuron in the layer is connected to every neuron in the previous and subsequent layers. These layers are commonly used in traditional feedforward neural networks.
4. Flatten: The Flatten layer is used to reshape the data. It transforms multi-dimensional data, such as the output of convolutional layers, into a one-dimensional vector. This is often necessary when transitioning from convolutional layers to fully connected layers in a neural network.
5. Dropout: Dropout is a regularization technique used in neural networks to prevent overfitting. The Dropout layer randomly sets a fraction of input units to zero during training. This helps to prevent the network from relying too heavily on specific neurons and encourages more robust and generalizable learning.

# Note
code : 
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True) 
    
This code is for avoid OOM error. OOM (Out Of Memory) errors can occur when building and training a neural network model on the GPU. The size of the model is limited by the available memory on the GPU. The following may occur when a model has exhausted the memory : Resource Exhausted Error : an error message that indicates Out Of Memory (OOM)
  

# Steps for building an image classification machine learning model
1. Data Collection
2. Data Preprocessing
3. Split Data
4. Model Selection - CNNs are widely used and highly effective for this task.
5. Model Training
6. Hyperparameter Tuning
7. Evaluation
8. Testing
