package nn.tensor.initialiser;

import nn.tensor.Tensor;

public class RandomInitialiser extends Initialiser {

    public RandomInitialiser(){
        super();
    }

    public void initializeTensor(Tensor tensor) {
        for (int i = 0; i < tensor.shape.volume; i++){
            tensor.elements[i] = random.nextFloat() * 2 - 1;
        }
    }
}
