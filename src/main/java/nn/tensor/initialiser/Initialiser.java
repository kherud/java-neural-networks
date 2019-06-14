package nn.tensor.initialiser;

import nn.tensor.Tensor;

import java.util.Random;

public abstract class Initialiser {

    Random random;

    Initialiser(){
        this.random = new Random(System.nanoTime());
    }

    public abstract void initializeTensor(Tensor tensor);
}
