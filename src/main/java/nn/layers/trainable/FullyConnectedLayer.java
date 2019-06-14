package nn.layers.trainable;

import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.error.ShapeMismatchException;
import nn.tensor.initialiser.Initialiser;

import java.util.stream.IntStream;

public class FullyConnectedLayer extends TrainableLayer {

    private final boolean stream;

    public FullyConnectedLayer(Shape inputShape, Shape outputShape, Initialiser initialiser) {
        super(initialiser, inputShape, outputShape);
        weights = new Tensor(new Shape(outputShape, inputShape));
        bias = new Tensor(outputShape);
        super.initWeights();
        super.initBias();
        stream = inputShape.volume > 200 && outputShape.volume > 200;
    }

    public void forward(Tensor inTensor, Tensor outTensor) {
        if (stream) {
            IntStream.range(0, outputShape.dimension[0]).parallel().forEach(i -> {
                outTensor.elements[i] = bias.elements[i];
                for (int j = 0; j < inputShape.dimension[0]; j++) {
                    outTensor.elements[i] += inTensor.elements[j] * weights.elements[i * inputShape.dimension[0] + j];
                }
            });
        } else {
            for (int i = 0; i < outputShape.dimension[0]; i++) {
                outTensor.elements[i] = bias.elements[i];
                for (int j = 0; j < inputShape.dimension[0]; j++) {
                    outTensor.elements[i] += inTensor.elements[j] * weights.elements[i * inputShape.dimension[0] + j];
                }
            }
        }
    }


    public void backward(Tensor outTensor, Tensor inTensor) {
        if (stream) {
            IntStream.range(0, inputShape.dimension[0]).parallel().forEach(i -> {
                outTensor.delta[i] = 0;
                for (int j = 0; j < outputShape.dimension[0]; j++) {
                    outTensor.delta[i] += inTensor.delta[j] * weights.elements[inputShape.dimension[0] * j + i];
                }
            });
        } else {
            for (int i = 0; i < inputShape.dimension[0]; i++) {
                outTensor.delta[i] = 0;
                for (int j = 0; j < outputShape.dimension[0]; j++) {
                    outTensor.delta[i] += inTensor.delta[j] * weights.elements[inputShape.dimension[0] * j + i];
                }
            }
        }
    }

    public void calculateDeltaWeights(Tensor inTensor, Tensor outTensor) {
        for (int i = 0; i < outputShape.dimension[0]; i++) {
            bias.delta[i] += inTensor.delta[i];
            for (int j = 0; j < inputShape.dimension[0]; j++) {
                weights.delta[inputShape.dimension[0] * i + j] += inTensor.delta[i] * outTensor.elements[j];
            }
        }
    }

    @Override
    public String toString() {
        return "Fully Connected Layer\n" + super.toString();
    }
}
