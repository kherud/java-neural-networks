package nn.layers.trainable;

import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.layers.Layer;
import nn.tensor.initialiser.Initialiser;

public abstract class TrainableLayer extends Layer {
    public Tensor weights;
    public Tensor bias;

    private Initialiser initialiser;

    TrainableLayer(Initialiser initialiser, Shape inputShape, Shape outputShape){
        super(inputShape, outputShape);
        this.initialiser = initialiser;
    }

    TrainableLayer(Initialiser initialiser, Shape inputShape){
        super(inputShape);
        this.initialiser = initialiser;
    }

    public abstract void calculateDeltaWeights(Tensor inTensor, Tensor outTensor);

    public int countParameters(){
        return weights.shape.volume + bias.shape.volume;
    }

    void initWeights(){
        initialiser.initializeTensor(weights);
    }

    void initBias(){
        initialiser.initializeTensor(bias);
    }

    @Override
    public String toString(){
        return super.toString() + "\tTrainable Parameters: " + countParameters();
    }
}
