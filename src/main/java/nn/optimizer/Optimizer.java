package nn.optimizer;

import nn.NeuralNetwork;
import nn.tensor.Tensor;
import nn.layers.Layer;
import nn.layers.trainable.TrainableLayer;
import nn.loss.Loss;

public abstract class Optimizer {

    private Loss lossFunction;
    private int batchSize;
    int epoch;

    Optimizer(Loss lossFunction, int batchSize) {
        this.lossFunction = lossFunction;
        this.batchSize = batchSize;
    }

    public abstract void updateParameter(Tensor parameter, int index);

    public void optimize(NeuralNetwork nn, Tensor[] trainX, Tensor[] trainY, Tensor[] testX, Tensor[] testY, int epochs) {
        System.out.println("Start Training...");
        for (int epoch = 0; epoch < epochs; epoch++) {
            this.epoch = epoch;
            nn.state = NeuralNetwork.State.TRAIN;

            float loss = 0;
            System.out.println("Epoch " + (epoch + 1));

            for (int batch = 0; batch < trainX.length / batchSize; batch++) {
                for (int i = batch * batchSize; i < (batch + 1) * batchSize; i++) {
                    Tensor prediction = nn.forward(trainX[i]);
                    lossFunction.reset();
                    lossFunction.forward(prediction, trainY[i]);

                    loss -= loss / (i + 1) - lossFunction.loss.elements[0] / (i + 1);

                    lossFunction.backward(prediction, trainY[i]);
                    nn.backward(trainX[i]);

                    System.out.print("\r" + (i + 1) + "/" + trainX.length + " loss: " + loss);
                }
                updateParameters(nn);
            }

            System.out.println();

            if (testX != null && testY != null) {
                nn.state = NeuralNetwork.State.PREDICT;

                System.out.print("evaluating ...");
                Tensor evaluation = nn.evaluate(testX, testY, lossFunction);
                System.out.print(" val_loss: " + evaluation.elements[0] + " - val_acc: " + evaluation.elements[1]);
                System.out.println();
            }
        }
    }


    private void updateParameters(NeuralNetwork nn){
        for (Layer layer : nn.layers) {
            if (!(layer instanceof TrainableLayer))
                continue;

            Tensor weights = ((TrainableLayer) layer).weights;
            for (int w = 0; w < weights.shape.volume; w++) {
                updateParameter(weights, w);
            }
            weights.resetDeltas();

            Tensor bias = ((TrainableLayer) layer).bias;
            for (int b = 0; b < bias.shape.volume; b++) {
                updateParameter(bias, b);
            }
            bias.resetDeltas();
        }
    }
}