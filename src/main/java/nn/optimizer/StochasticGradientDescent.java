package nn.optimizer;

import nn.tensor.Tensor;
import nn.loss.Loss;

public class StochasticGradientDescent extends Optimizer {

    private float learningRate;
    private float decay;

    public StochasticGradientDescent(Loss lossFunction, int batchSize, float learningRate, float decay){
        super(lossFunction, batchSize);
        this.learningRate = learningRate;
        this.decay = decay;
    }

    public StochasticGradientDescent(Loss lossFunction, int batchSize, float learningRate){
        super(lossFunction, batchSize);
        this.learningRate = learningRate;
        this.decay = 0;
    }

    @Override
    public void updateParameter(Tensor parameter, int index){
        parameter.elements[index] -= (learningRate * (1 / (1 + decay * epoch)) * parameter.delta[index]);
    }
}
