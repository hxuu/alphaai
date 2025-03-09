# Ergo Proxy

First time solving an AI challenge.

## Analysis (First Dive)

I downloaded the data, now let's check what it is:

1. PyTorch model (.pth file)
2. A jupyter notebook containing the code used to train a model

Words needing explanation: PyTorch. Jupyter Notebook. Training a model

a. PyTorch is an open source deep learning framework used for building and training
neural networks.
b. Jupyter notebook is an online medium to run and execute python code
c. Training a model consists of teaching a machine learning algorith to make predictions
and classify data.

- By understanding the training process, you can reverse-engineer the model to
identify the flag image (label 1).

### Concepts Related to this challenge

1. Neural networks: It's a machine learning model inspired by the human brain.

The intuition is that our human brain has a huge recognition power. an obfsucated
or pixelated THREE can be easily recognized, so how? The answer is simply by neural
networks.

- What are Neurons? -> A thing that holds a number (between 0 and 1)

Neurons are linked togther and make a `Neural Network` in the shape of layers.
We start by a first layer containing the raw image, the way that image is constructed,
i.e which pixels do fire will trigger some neurons in the next layer to be fired.
Ultimately, the last layer will contain 10 neurons each indicating a digit. They will
all fire with different `activation numbers` indicating how close they are to the
actual number represented by the image.

The way those neurons are constructed is by identifying certain features of the image
being fed to the model. Through some fancy math, the edges are given a weight to detemine
the value of whether the node should be lighted or not with the help of the sigmoid function
that squashes the weight sum of the activation numbers of the neurons in the last
layer and their edge weights in a range between `[0-1]`.

The neuron should be biased towards being inactive by adding a value (-10 in the example)
that acts like a threshold for the neuron to be activated.

- So when we talk about learning -> We're refering to find the right weights and biases
that make the neurons either fire or stay inactive. (That's the transition between layers!)
