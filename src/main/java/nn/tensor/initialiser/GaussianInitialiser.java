package nn.tensor.initialiser;

import nn.tensor.Tensor;

public class GaussianInitialiser extends Initialiser {
    @Override
    public void initializeTensor(Tensor tensor) {
        for (int i = 0; i < tensor.elements.length; i++) {
            tensor.elements[i] = (float) random.nextGaussian() * 2 / tensor.shape.volume;
        }
    }
}
