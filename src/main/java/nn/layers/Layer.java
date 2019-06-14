package nn.layers;

import nn.tensor.Shape;
import nn.tensor.Tensor;

public abstract class Layer {

    public Shape inputShape;
    public Shape outputShape;

    public Layer(Shape inputShape, Shape outputShape){
        this.inputShape = inputShape;
        this.outputShape = outputShape;
    }

    public Layer(Shape inputShape){
        this.inputShape = inputShape;
    }

    public abstract void forward(Tensor inTensor, Tensor outTensor);
    public abstract void backward(Tensor outTensor, Tensor inTensor);

    @Override
    public String toString(){
        return "\tInput Shape: " + inputShape + " - " +
                "Output Shape: " + outputShape + "\n";
    }
}
