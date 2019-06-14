package nn.optimizer;

import nn.tensor.Tensor;
import nn.loss.Loss;

public class SimpleAdam extends Optimizer {

    private float learningRate;
    private float beta1 = 0.9f, beta2 = 0.999f;
    private float m = 0, v = 0;
    private float epsilon = 1e-8f;

    public SimpleAdam(Loss lossFunction, int batchSize, float learningRate){
        super(lossFunction, batchSize);
        this.learningRate = learningRate;
    }

    @Override
    public void updateParameter(Tensor parameter, int index){
        m = beta1 * m + (1 - beta1) * parameter.delta[index];
        v = beta2 * v + (1 - beta2) * (parameter.delta[index] * parameter.delta[index]);

        parameter.elements[index] -= (learningRate * m / (Math.sqrt(v) + epsilon));
    }
}
