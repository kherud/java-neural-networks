package nn.layers.activation;

import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.error.ShapeMismatchException;

public class ReLULayer extends ActivationLayer {
    public ReLULayer(Shape shape) {
        super(shape);
    }

    public void forward(Tensor inTensor, Tensor outTensor) {
        for (int i = 0; i < outputShape.dimension[0]; i++){
            outTensor.elements[i] = inTensor.elements[i] > 0 ? inTensor.elements[i] : 0;
        }
    }

    public void backward(Tensor outTensor, Tensor inTensor) {
        for (int i = 0; i < inputShape.dimension[0]; i++){
            outTensor.delta[i] = outTensor.elements[i] > 0 ? inTensor.delta[i] : 0;
        }
    }

    @Override
    public String toString() {
        return "ReLU Layer (Activation)";
    }
}
