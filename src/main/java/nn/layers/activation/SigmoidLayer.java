package nn.layers.activation;

import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.error.ShapeMismatchException;

public class SigmoidLayer extends ActivationLayer {

    public SigmoidLayer(Shape shape) {
        super(shape);
    }

    public void forward(Tensor inTensor, Tensor outTensor) {
        for (int i = 0; i < outputShape.dimension[0]; i++) {
            outTensor.elements[i] = 1 / (1 + (float) Math.exp(-inTensor.elements[i]));
        }
    }

    public void backward(Tensor outTensor, Tensor inTensor) {
        for (int i = 0; i < inputShape.dimension[0]; i++) {
            outTensor.delta[i] = (inTensor.elements[i] * (1 - inTensor.elements[i])) * inTensor.delta[i];
        }
    }

    public String toString() {
        return "Sigmoid Layer (Activation)";
    }
}
