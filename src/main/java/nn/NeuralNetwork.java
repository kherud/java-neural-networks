package nn;

import nn.error.ShapeMismatchException;
import nn.layers.Layer;
import nn.layers.trainable.TrainableLayer;
import nn.layers.utility.Dropout;
import nn.loss.Loss;
import nn.tensor.Shape;
import nn.tensor.Tensor;

import java.util.Arrays;
import java.util.stream.IntStream;

public class NeuralNetwork {

    public enum State {
        TRAIN, PREDICT
    }

    public NeuralNetwork.State state;

    public Layer[] layers;
    public Tensor[] tensors;

    public NeuralNetwork(Layer... layers) {
        this.layers = layers;
        this.state = NeuralNetwork.State.TRAIN;

        tensors = new Tensor[layers.length];
        for (int i = 0; i < layers.length; i++) {
            tensors[i] = new Tensor(layers[i].outputShape);
        }

        compile();
    }

    public Tensor forward(Tensor inTensor) {
        for (int i = 0; i < layers.length; i++) {
            if (state == State.PREDICT && layers[i] instanceof Dropout)
                continue;

            if (!inTensor.shape.equals(layers[i].inputShape) && inTensor.shape.volume == layers[i].inputShape.volume)
                inTensor.reshape(layers[i].inputShape);
            if (!tensors[i].shape.equals(layers[i].outputShape) && tensors[i].shape.volume == layers[i].outputShape.volume)
                tensors[i].reshape(layers[i].outputShape);

            layers[i].forward(inTensor, tensors[i]);
            inTensor = tensors[i];
        }
        return inTensor;
    }

    public void backward(Tensor inTensor) {
        if (state == State.PREDICT)
            throw new IllegalStateException("Backward pass in prediction state");

        for (int i = layers.length - 1; i > 0; i--) {
            if (!tensors[i].shape.equals(layers[i].outputShape) && tensors[i].shape.volume == layers[i].outputShape.volume)
                tensors[i].reshape(layers[i].outputShape);

            layers[i].backward(tensors[i - 1], tensors[i]);
        }

        calculateDeltaWeights(inTensor);
    }

    private void calculateDeltaWeights(Tensor inTensor) {
        IntStream.range(0, layers.length).parallel().forEach(
                layer -> {
                    if (!(layers[layer] instanceof TrainableLayer))
                        return;
                    if (layer == 0) ((TrainableLayer) layers[layer]).calculateDeltaWeights(tensors[layer], inTensor);
                    else ((TrainableLayer) layers[layer]).calculateDeltaWeights(tensors[layer], tensors[layer - 1]);
                });
    }

    public Tensor evaluate(Tensor[] xTensors, Tensor[] yTensors, Loss lossFunction) {
        Tensor evaluation = new Tensor(new Shape(2));

        for (int i = 0; i < xTensors.length; i++) {
            Tensor prediction = forward(xTensors[i]);

            lossFunction.reset();
            lossFunction.forward(prediction, yTensors[i]);

            // check if loss is NaN (this happens by multiplication with Inf, e.g. by log(0) in crossentropy)
            if (lossFunction.loss.elements[0] == lossFunction.loss.elements[0])
                evaluation.elements[0] -= evaluation.elements[0] / (i + 1) - lossFunction.loss.elements[0] / (i + 1);

            int indexPredMax = 0, indexLabelMax = 0;
            for (int j = 0; j < prediction.elements.length; j++) {
                if (prediction.elements[j] > prediction.elements[indexPredMax]) {
                    indexPredMax = j;
                }
                if (yTensors[i].elements[j] > yTensors[i].elements[indexLabelMax]) {
                    indexLabelMax = j;
                }
            }
            float predictionCorrect = indexPredMax == indexLabelMax ? 1 : 0;
            evaluation.elements[1] -= evaluation.elements[1] / (i + 1) - predictionCorrect / (i + 1);

            System.out.print("\r" + (i + 1) + "/" + xTensors.length);
        }

        return evaluation;
    }

    public void summary() {
        String fullDelimiter = String.format("\n%0" + 65 + "d\n", 0).replace("0", "=");
        String halfDelimiter = String.format("\n%0" + 65 + "d\n", 0).replace("0", "-");
        StringBuilder summary = new StringBuilder();
        summary.append(fullDelimiter).append("Neural Network Summary").append(fullDelimiter);
        for (int i = 0; i < layers.length - 1; i++) {
            summary.append(layers[i].toString()).append(halfDelimiter);
        }
        summary.append(layers[layers.length - 1]).append(fullDelimiter);
        int parameterSum = Arrays.stream(layers)
                .filter(l -> l instanceof TrainableLayer)
                .map(l -> (TrainableLayer) l)
                .mapToInt(TrainableLayer::countParameters).sum();
        summary.append("Total trainable tensors: ").append(parameterSum).append(fullDelimiter);
        System.out.println(summary.toString());
    }

    private void compile() {
        if (layers.length == 0)
            throw new IllegalStateException("No layers were added to the neural network");

        Tensor inTensor = new Tensor(layers[0].inputShape);
        for (int i = 0; i < layers.length; i++) {
            if (inTensor.shape.volume != layers[i].inputShape.volume)
                throw new ShapeMismatchException("Error with input shape in layer " + i + " ("
                        + layers[i].getClass().getSimpleName() + ")");

            inTensor = tensors[i];
        }
    }
}
