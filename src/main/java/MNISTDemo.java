import data.access.MNISTLoader;
import nn.NeuralNetwork;
import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.layers.activation.SoftmaxLayer;
import nn.tensor.initialiser.XavierInitialiser;
import nn.layers.trainable.FullyConnectedLayer;
import nn.layers.activation.SigmoidLayer;
import nn.loss.CrossEntropy;
import nn.optimizer.Optimizer;
import nn.optimizer.StochasticGradientDescent;


public class MNISTDemo {

    public static void main(String[] args) {
        Shape inputShape = new Shape(784);

        NeuralNetwork nn = new NeuralNetwork(
                new FullyConnectedLayer(
                        inputShape,
                        new Shape(256), // output shape
                        new XavierInitialiser()),
                new SigmoidLayer(new Shape(256)),
                // new Dropout(new Shape(256), 0.5f),
                new FullyConnectedLayer(
                        new Shape(256),
                        new Shape(64),
                        new XavierInitialiser()),
                new SigmoidLayer(new Shape(64)),
                // new Dropout(new Shape(64), 0.5f),
                new FullyConnectedLayer(
                        new Shape(64),
                        new Shape(10),
                        new XavierInitialiser()),
                new SoftmaxLayer(new Shape(10))
        );

        nn.summary();

        MNISTLoader dataLoader = new MNISTLoader();
        Tensor[][] trainData = dataLoader.loadTrain(inputShape);
        Tensor[] trainX = trainData[0];
        Tensor[] trainY = trainData[1];

        Tensor[][] testData = dataLoader.loadTest(inputShape);
        Tensor[] testX = testData[0];
        Tensor[] testY = testData[1];

        Optimizer optimizer = new StochasticGradientDescent(new CrossEntropy(), 32,0.1f, 0.1f);
        optimizer.optimize(nn, trainX, trainY, testX, testY, 10);
    }
}
