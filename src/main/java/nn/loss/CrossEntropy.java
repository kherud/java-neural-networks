package nn.loss;

import nn.tensor.Tensor;
import nn.error.ShapeMismatchException;

public class CrossEntropy extends Loss {

    public CrossEntropy() {
        super();
    }

    public void forward(Tensor prediction, Tensor truth) {
        if (!prediction.shape.equals(truth.shape))
            throw new ShapeMismatchException("Shapes did not match in forward pass");

        for (int i = 0; i < prediction.elements.length; i++) {
            loss.elements[0] += truth.elements[i] * Math.log(prediction.elements[i]);
        }
        loss.elements[0] *= -1;
    }

    public void backward(Tensor prediction, Tensor truth) {
        if (!prediction.shape.equals(truth.shape))
            throw new ShapeMismatchException("Shapes did not match in backward pass");

        for (int i = 0; i < prediction.elements.length; i++) {
            prediction.delta[i] = -truth.elements[i] / prediction.elements[i];
        }
    }
}
