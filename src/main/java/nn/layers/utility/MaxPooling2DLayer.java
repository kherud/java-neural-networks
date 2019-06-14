package nn.layers.utility;

import nn.tensor.Shape;
import nn.tensor.Tensor;

public class MaxPooling2DLayer extends PoolingLayer {

    public MaxPooling2DLayer(Shape inputShape, Shape pooling, Shape stride) {
        super(inputShape, pooling, stride);
    }

    @Override
    void pool(Tensor inTensor, Tensor outTensor, int channel, int x, int y){
        int outIndex = channel * (outputShape.dimension[0] * outputShape.dimension[1])
                + y / stride.dimension[1] * outputShape.dimension[0]
                + x / stride.dimension[1];
        float max = Float.NEGATIVE_INFINITY;
        int maxIndex = 0;
        for (int i = 0; i < pooling.dimension[0]; i++){
            for (int j = 0; j < pooling.dimension[1]; j++){
                int pixelIndex = channel * (inputShape.dimension[0] * inputShape.dimension[1])
                        + (y + j) * inputShape.dimension[0] + (x + i);
                if (inTensor.elements[pixelIndex] > max){
                    outTensor.elements[outIndex] = max = inTensor.elements[pixelIndex];
                    maxIndex = pixelIndex;
                }
            }
        }
        poolingMask[maxIndex] = outIndex;
    }

    @Override
    public String toString() {
        return "2D Max Pooling Layer\n" + super.toString();
    }
}
