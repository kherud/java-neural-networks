package nn.layers.utility;

import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.error.ShapeMismatchException;
import nn.layers.Layer;

import java.util.Random;

public class Dropout extends Layer {

    private float rate;

    public float[] mask;
    private final float[] zeroBuffer;
    private final Random random;

    public Dropout(Shape inputShape, float dropoutRate){
        super(inputShape, inputShape);
        this.rate = dropoutRate;
        this.mask = new float[inputShape.volume];
        this.zeroBuffer = new float[inputShape.volume];
        this.random = new Random();
    }

    public Dropout(Shape inputShape, float dropoutRate, Random random){
        super(inputShape, inputShape);
        this.rate = dropoutRate;
        this.mask = new float[inputShape.volume];
        this.zeroBuffer = new float[inputShape.volume];
        this.random = random;
    }

    @Override
    public void forward(Tensor inTensor, Tensor outTensor) {
        for (int i = 0; i < inputShape.volume; i++){
            this.mask[i] = random.nextFloat() > rate ? inTensor.elements[i] : 0;
            outTensor.elements[i] = mask[i];
        }
    }

    @Override
    public void backward(Tensor outTensor, Tensor inTensor) {
        for (int i = 0; i < outputShape.volume; i++){
            outTensor.delta[i] = mask[i] == 0 ? 0 : inTensor.delta[i];
        }
        System.arraycopy(zeroBuffer, 0, mask, 0, zeroBuffer.length);
    }

    @Override
    public String toString() {
        return "Dropout Layer (Dropout " + rate + ")";
    }
}
