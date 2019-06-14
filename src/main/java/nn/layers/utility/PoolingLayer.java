package nn.layers.utility;

import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.error.InputShapeException;
import nn.layers.Layer;

import java.util.stream.IntStream;

public abstract class PoolingLayer extends Layer {

    Shape stride;
    Shape pooling;

    public int[] poolingMask;

    PoolingLayer(Shape inputShape, Shape pooling, Shape stride) {
        super(inputShape);
        this.stride = stride;
        this.pooling = pooling;
        this.outputShape = calculateOutputShape();
        poolingMask = new int[inputShape.volume];
        resetPoolingMask();
    }

    abstract void pool(Tensor inTensor, Tensor outTensor, int channel, int x, int y);

    @Override
    public void forward(Tensor inTensor, Tensor outTensor) {
        IntStream.range(0, inputShape.dimension[2]).parallel().forEach(
            channel -> {
                for (int x = 0; x <= inputShape.dimension[0] - pooling.dimension[0]; x += stride.dimension[0]){
                    for (int y = 0; y <= inputShape.dimension[1] - pooling.dimension[1]; y += stride.dimension[1]){
                        pool(inTensor, outTensor, channel, x, y);
                    }
                }
            });
    }

    @Override
    public void backward(Tensor outTensor, Tensor inTensor) {
        for (int i = 0; i < outTensor.delta.length; i++){
            outTensor.delta[i] = poolingMask[i] == -1 ? 0 : inTensor.delta[poolingMask[i]];
        }
        resetPoolingMask();
    }

    private void resetPoolingMask(){
        for (int i = 0; i < inputShape.volume; i++){
            poolingMask[i] = -1;
        }
    }

    private Shape calculateOutputShape() {
        if (inputShape.dimension.length != 3) {
            throw new InputShapeException("Pooling Layer has wrong input shape, needed: (x, y, n_channels)");
        }
        if (pooling.dimension.length != 2) {
            throw new InputShapeException("Invalid pooling shape, needed: (x, y)");
        }

        int oX = (inputShape.dimension[0] - pooling.dimension[0]) / stride.dimension[0] + 1;
        int oY = (inputShape.dimension[1] - pooling.dimension[1]) / stride.dimension[1] + 1;

        return new Shape(oX, oY, inputShape.dimension[2]);
    }

    @Override
    public String toString(){
        return "\tPooling: " + pooling
                + " - Stride: " + stride + "\n"
                + super.toString();
    }
}

