package nn.tensor.initialiser;

import nn.tensor.Tensor;

public class ZeroInitialiser extends Initialiser {
    /**
     * Redundant since arrays are initialised with 0 anyway
     */
    public ZeroInitialiser(){
        super();
    }

    public void initializeTensor(Tensor tensor) {
        for (int i = 0; i < tensor.shape.volume; i++){
            tensor.elements[i] = 0;
        }
    }
}
