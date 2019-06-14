package nn.layers.activation;

import nn.tensor.Shape;
import nn.layers.Layer;

public abstract class ActivationLayer extends Layer {

    ActivationLayer(Shape shape){
        super(shape, shape);
    }

}
