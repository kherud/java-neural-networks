package nn.loss;

import nn.tensor.Shape;
import nn.tensor.Tensor;

public abstract class Loss {

    public Tensor loss;

    Loss(){
        loss = new Tensor(new Shape(1));
        loss.delta[0] = 1;
    }

    public abstract void forward(Tensor prediction, Tensor truth);
    public abstract void backward(Tensor prediction, Tensor truth);

    public void reset(){
        loss.elements[0] = 0;
    }
}
