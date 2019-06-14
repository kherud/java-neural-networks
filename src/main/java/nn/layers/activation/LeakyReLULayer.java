package nn.layers.activation;

import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.error.ShapeMismatchException;

public class LeakyReLULayer extends ActivationLayer {

    private float rate;

    public LeakyReLULayer(Shape shape, float rate) {
        super(shape);
        this.rate = rate;
    }

    public LeakyReLULayer(Shape shape) {
        this(shape, 0.01f);
    }

    @Override
    public void forward(Tensor inTensor, Tensor outTensor) {
        for (int i = 0; i < outputShape.dimension[0]; i++) {
            outTensor.elements[i] = inTensor.elements[i] > 0 ? inTensor.elements[i] : rate * inTensor.elements[i];
        }
    }

    @Override
    public void backward(Tensor outTensor, Tensor inTensor) {
        for (int i = 0; i < inputShape.dimension[0]; i++){
            outTensor.delta[i] = outTensor.elements[i] > 0 ? inTensor.delta[i] : rate * inTensor.delta[i];
        }
    }

    @Override
    public String toString() {
        return "Leaky (" + rate + ") ReLU Layer (Activation)";
    }

}
