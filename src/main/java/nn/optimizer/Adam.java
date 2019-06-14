package nn.optimizer;

import nn.tensor.Tensor;
import nn.loss.Loss;

public class Adam extends Optimizer {
    private float learningRate;
    private float beta1 = 0.9f, beta2 = 0.999f;
    private float m = 0, v = 0, t = 1;
    private float epsilon = 1e-8f;

    public Adam(Loss lossFunction, int batchSize, float learningRate) {
        super(lossFunction, batchSize);
        this.learningRate = learningRate;
    }

    @Override
    public void updateParameter(Tensor parameter, int index) {
        m = beta1 * m + (1 - beta1) * parameter.delta[index];
        double mt = m / (1 - Math.pow(beta1, t));

        v = beta2 * v + (1 - beta2) * (parameter.delta[index] * parameter.delta[index]);
        double vt = v / (1 - Math.pow(beta2, t));

        parameter.elements[index] -= learningRate * mt / (Math.sqrt(vt) + epsilon);
    }
}
