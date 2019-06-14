package nn.tensor.initialiser;

import nn.tensor.Tensor;

public class XavierInitialiser extends Initialiser {

    public XavierInitialiser(){
        super();
    }

    public void initializeTensor(Tensor tensor) {
        for (int i = 0; i < tensor.shape.volume; i++){
            tensor.elements[i] = (random.nextFloat() * 2 - 1) / (float) Math.sqrt(tensor.shape.volume);
        }
    }
}
