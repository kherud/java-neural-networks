package nn.loss;

import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.error.ShapeMismatchException;

public class MeanSquaredError extends Loss {

    public MeanSquaredError(){
        super();
    }

    public void forward(Tensor prediction, Tensor truth) {
        if (!prediction.shape.equals(truth.shape))
            throw new ShapeMismatchException("Shapes did not match in forward pass");

        Tensor loss = new Tensor(new Shape(1));
        for (int i = 0; i < prediction.elements.length; i++){
            loss.elements[0] += (prediction.elements[i] - truth.elements[i]) * (prediction.elements[i] - truth.elements[i]);
        }
        loss.elements[0] *= .5f;
    }

    public void backward(Tensor prediction, Tensor truth) {
        if (!prediction.shape.equals(truth.shape))
            throw new ShapeMismatchException("Shapes did not match in backward pass");

        for (int i = 0; i < prediction.elements.length; i++){
            prediction.delta[i] = truth.elements[i] - prediction.elements[i];
        }
    }
}
