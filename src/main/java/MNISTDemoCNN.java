import data.access.MNISTLoader;
import nn.NeuralNetwork;
import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.layers.activation.ReLULayer;
import nn.layers.activation.SigmoidLayer;
import nn.layers.activation.SoftmaxLayer;
import nn.tensor.initialiser.GaussianInitialiser;
import nn.tensor.initialiser.XavierInitialiser;
import nn.layers.trainable.Conv2DLayer;
import nn.layers.trainable.FullyConnectedLayer;
import nn.layers.utility.Dropout;
import nn.layers.utility.MaxPooling2DLayer;
import nn.loss.CrossEntropy;
import nn.optimizer.Optimizer;
import nn.optimizer.StochasticGradientDescent;


public class MNISTDemoCNN {

    public static void main(String[] args) {
        Shape inputShape = new Shape(28, 28, 1);

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
//                new Conv2DLayer(new Shape(8, 8, 18),
//                        new Shape(3, 3, 6),
//                        new Shape(1, 1),
//                        Conv2DLayer.Padding.NONE,
//                        new GaussianInitialiser()),
//                new ReLULayer(new Shape(6 * 6 * 6)),
                new FullyConnectedLayer(new Shape(8 * 8 * 18),
                        new Shape(120),
                        new XavierInitialiser()),
                new SigmoidLayer(new Shape(120)),
                new Dropout(new Shape(120), 0.25f),
                new FullyConnectedLayer(new Shape(120),
                        new Shape(45),
                        new XavierInitialiser()),
                new SigmoidLayer(new Shape(45)),
                new Dropout(new Shape(45), 0.25f),
                new FullyConnectedLayer(new Shape(45),
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

        Optimizer optimizer = new StochasticGradientDescent(new CrossEntropy(), 32, 0.1f, 0.15f);
        optimizer.optimize(nn, trainX, trainY, testX, testY, 10);
    }
}
