# Java Neural Network Library

[![Build Status](https://travis-ci.com/kherud/java-neural-networks.svg?branch=master)](https://travis-ci.com/kherud/java-neural-networks)

This is a small deep learning framework mainly implemented to learn how things work.

There are fully connected and convolutional layers.

Tensors are memory optimized.

```
NeuralNetwork nn = new NeuralNetwork(
        new Conv2DLayer(inputShape, // input shape
                new Shape(3, 3, 18), // filter size
                new Shape(1, 1), // stride
                Conv2DLayer.Padding.NONE, // padding
                new GaussianInitialiser(),
                false),
        new ReLULayer(new Shape(26 * 26 * 18)),
        new MaxPooling2DLayer(
                new Shape(26, 26, 18), // input shape
                new Shape(3, 3), // pooling shape
                new Shape(3, 3) // stride
        ),
        new Dropout(new Shape(8 * 8 * 18), 0.25f),
        new FullyConnectedLayer(new Shape(8 * 8 * 18),
                new Shape(120),
                new XavierInitialiser()),
        new SigmoidLayer(new Shape(120)),
        new Dropout(new Shape(120), 0.25f),
        new FullyConnectedLayer(new Shape(120),
                new Shape(10),
                new XavierInitialiser()),
        new SoftmaxLayer(new Shape(10))
);
```
