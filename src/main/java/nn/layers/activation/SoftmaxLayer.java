package nn.layers.activation;

import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.error.ShapeMismatchException;

public class SoftmaxLayer extends ActivationLayer {

    public SoftmaxLayer(Shape shape) {
        super(shape);
    }

    public void forward(Tensor inTensor, Tensor outTensor) {
        float sum = 0;
        for (int i = 0; i < inTensor.shape.dimension[0]; i++) {
            sum += Math.exp(inTensor.elements[i]);
        }

        for (int i = 0; i < outputShape.dimension[0]; i++) {
            outTensor.elements[i] = (float) Math.exp(inTensor.elements[i]) / sum;
        }
    }

    public void backward(Tensor outTensor, Tensor inTensor) {
        for (int i = 0; i < outputShape.dimension[0]; i++) {
            outTensor.delta[i] = 0;
            for (int j = 0; j < outputShape.dimension[0]; j++) {
                if (i == j)
                    outTensor.delta[i] += inTensor.delta[j] * (inTensor.elements[i] * (1 - inTensor.elements[i]));
                else
                    outTensor.delta[i] += inTensor.delta[j] * (-inTensor.elements[j] * inTensor.elements[i]);
            }
        }
    }

    @Override
    public String toString() {
        return "Softmax Layer (Activation)";
    }
}
