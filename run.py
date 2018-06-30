from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Stochastic data is commonly seeded with random integers
np.random.seed(7)

# Dataset (csv)
dataset = np.loadtxt("data.csv", delimiter=",")

# Split into input (X) and output (Y) variables
X = dataset[:,0:8]  # Columns 0-7
Y = dataset[:,8]  # Column 8

# Training a network means finding the best set of wights to make a prediction for this problem.

#  Define model
#   We create a sequential model and add layers one at a time until we are happy with network topology.
#   Firstly, ensure the input layer has correct number of outputs - specified with input_dim arg.
#
#  How to determine number of layers and their types?
#   It's hard. There are heuristics which can be used. Often the best network structure is found through
#   a process of trial and error / experimentation. Generally, a guideline is that the network needs to
#   large enough to capture the structure of the problem.
#
#  In this example, we're using a "fully connected network" with 3 layers.
#   Fully connected layers are defined using the "Dense" class. Using three inputs, we can specify the
#   number of neurons in a layer, the networks' "initialization method", and the "activation function".
#
#  Furthermore in this example, we initialize the network "weights" to be a small random number generated
#   from a "uniform distribution" - between 0 and 0.05 in this case as that's the default UD in keras.
#   Another traditional alternative would be "normal" for small random numbers generated from a gaussian
#   distribution.
#
#  Rectifier - ReLU (Rectifier linear unit). Currently the most popular activation function in DL. This is the
#   AF we'll be using on the first two layers. and the sigmoid in the output layer. Using the ReLU AF, better
#   performance can be achieved than an older convention which was to use Sigmoid and tanh AFs for all layers.
#
#  The sigmoid on the output layer to ensure the network's output is between 0 and 1 and easy to map to either
#   a probability class 1 or snap to a hard classification of either class with a default threshold of 0.5.
#
#  We can piece it all together by adding each layer. The 1st layer has 12 neurons and expects 8 input variables.
#   The second layer has 8 neurons and finally, the output layer has 1 neuron to predict the class (final prediction).

# Create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Model compilation is used in keras to increase efficiency by using numerical libraries such as Tensorflow.
#   This will automatically choose the best way to represent the network for training and making predictions to run
#   on your hardware, such as CPU or GPU or even distributed.
# When compiling, we must specify additional properties required for training the network.
#   We must specify the "loss function" to evaluate a set of weights, the "optimizer" used to search through
#   different weights for the network and any additional metrics we would like to collect and report during training.
# In this algorithm, we use "logarithmic loss", which for binary classifications problems in Keras is defined as
#   "binary_crossentropy".
# Because it's a classification problem, we will collect the classification accuracy.

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Now we have compiled the model to be more efficient, it's time to execute the model on some data.
#
# We train or fit the model on the loaded data by using the fit() function on the model.
#
# The training process will run for a fixed number of iterations through the dataset called "epochs",
#  that we must specify using the nepochs argument.

# Fit Model
