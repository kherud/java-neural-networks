package nn.layers.trainable;

import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.error.InputShapeException;
import nn.error.ShapeMismatchException;
import nn.tensor.initialiser.Initialiser;

import java.util.stream.IntStream;

public class Conv2DLayer extends TrainableLayer {
    public enum Padding {
        NONE, HALF, FULL
    }

    private Shape kernelShape;
    private Shape stride;
    private Padding padding;

    private int padX, padY;
    private int bPadX, bPadY;

    private boolean useBias;

    public Conv2DLayer(Shape inputShape, Shape kernelShape, Shape stride, Padding padding, Initialiser initialiser) {
        super(initialiser, inputShape);
        this.kernelShape = kernelShape;
        this.stride = stride;
        this.padding = padding;
        this.outputShape = calculateOutputShape();
        weights = new Tensor(new Shape(kernelShape, new Shape(inputShape.dimension[2])));
        bias = new Tensor(new Shape(kernelShape.dimension[2]));
        initialiser.initializeTensor(weights);
        initialiser.initializeTensor(bias);
        this.useBias = true;
    }

    public Conv2DLayer(Shape inputShape, Shape kernelShape, Shape stride, Padding padding, Initialiser initialiser, boolean useBias) {
        this(inputShape, kernelShape, stride, padding, initialiser);
        this.useBias = useBias;
    }


    @Override
    public void forward(Tensor inTensor, Tensor outTensor) {
        IntStream.range(0, outputShape.dimension[2]).parallel().forEach(
                filter -> {
                    for (int x = 0 - padX; x <= inputShape.dimension[0] + padX - kernelShape.dimension[0]; x += stride.dimension[0]) {
                        for (int y = 0 - padY; y <= inputShape.dimension[1] + padY - kernelShape.dimension[1]; y += stride.dimension[1]) {
                            forwardConvolute(inTensor, outTensor, filter, x, y);
                        }
                    }
                });
    }

    private void forwardConvolute(Tensor inTensor, Tensor outTensor, int filter, int x, int y) {
        int outIndex = filter * (outputShape.dimension[0] * outputShape.dimension[1])
                + ((y + padY) / stride.dimension[1]) * outputShape.dimension[0]
                + (x + padX) / stride.dimension[1];
        outTensor.elements[outIndex] = 0;
        for (int channel = 0; channel < inputShape.dimension[2]; channel++) {
            for (int i = 0; i < kernelShape.dimension[0]; i++) {
                for (int j = 0; j < kernelShape.dimension[1]; j++) {
                    if (x + i < 0 || x + i >= inputShape.dimension[0] || y + j < 0 || y + j >= inputShape.dimension[1]) {
                        continue; // padding -> multiplication with zero
                    }
                    int pixelIndex = channel * (inputShape.dimension[0] * inputShape.dimension[1])
                            + (y + j) * inputShape.dimension[0] + (x + i);
                    int kernelIndex = filter * (kernelShape.dimension[0] * kernelShape.dimension[1] * inputShape.dimension[2])
                            + channel * kernelShape.dimension[0] * kernelShape.dimension[1]
                            + j * kernelShape.dimension[0] + i;
                    outTensor.elements[outIndex] += inTensor.elements[pixelIndex] * weights.elements[kernelIndex];
                }
            }
        }
        if (useBias)
            outTensor.elements[outIndex] += bias.elements[filter];
    }

    @Override
    public void backward(Tensor outTensor, Tensor inTensor) {
        IntStream.range(0, inputShape.dimension[2]).parallel().forEach(
                channel -> {
                    for (int x = 0 - bPadX; x <= outputShape.dimension[0] + bPadX - kernelShape.dimension[0]; x += 1) {
                        for (int y = 0 - bPadY; y <= outputShape.dimension[1] + bPadY - kernelShape.dimension[1]; y += 1) {
                            backwardConvolute(outTensor, inTensor, channel, x, y);
                        }
                    }
                });
    }

    private void backwardConvolute(Tensor outTensor, Tensor inTensor, int channel, int x, int y) {
        int outIndex = channel * (inputShape.dimension[0] * inputShape.dimension[1])
                + (y + bPadY) * inputShape.dimension[0]
                + (x + bPadX);
        outTensor.delta[outIndex] = 0;
        for (int filter = 0; filter < kernelShape.dimension[2]; filter++) {
            for (int i = 0; i < kernelShape.dimension[0]; i++) {
                for (int j = 0; j < kernelShape.dimension[1]; j++) {
                    if (x + i < 0 || x + i >= outputShape.dimension[0] || y + j < 0 || y + j >= outputShape.dimension[1]) {
                        continue; // padding -> multiplication with zero
                    }
                    int pixelIndex = filter * (outputShape.dimension[0] * outputShape.dimension[1])
                            + (y + j) * outputShape.dimension[0] + (x + i);
                    int kernelIndex = filter * (kernelShape.dimension[0] * kernelShape.dimension[1] * inputShape.dimension[2])
                            + channel * kernelShape.dimension[0] * kernelShape.dimension[1]
                            + (kernelShape.dimension[0] * kernelShape.dimension[1] - 1 - j * kernelShape.dimension[0] - i); // shape - i -> rotation
                    outTensor.delta[outIndex] += inTensor.delta[pixelIndex] * weights.elements[kernelIndex];
                }
            }
        }
    }

    // TODO: parameter order got mixed up here - fix!
    @Override
    public void calculateDeltaWeights(Tensor outTensor, Tensor inTensor) {
        IntStream.range(0, outputShape.dimension[2]).parallel().forEach(
                filter -> {
                    for (int channel = 0; channel < inputShape.dimension[2]; channel++) {
                        calculateDeltaKernel(inTensor, outTensor, filter, channel);
                    }
                    if (useBias)
                        calculateDeltaBias(outTensor, filter);
                });
    }

    private void calculateDeltaBias(Tensor outTensor, int filter) {
        for (int i = 0; i < outputShape.dimension[0]; i++) {
            for (int j = 0; j < outputShape.dimension[1]; j++) {
                int deltaIndex = filter * (outputShape.dimension[0] * outputShape.dimension[1])
                        + j * outputShape.dimension[0] + i;
                bias.delta[filter] += outTensor.delta[deltaIndex];
            }
        }
    }

    private void calculateDeltaKernel(Tensor inTensor, Tensor outTensor, int filter, int channel) {
        for (int x = 0; x < kernelShape.dimension[0]; x++) {
            for (int y = 0; y < kernelShape.dimension[1]; y++) {
                convoluteDeltaKernel(inTensor, outTensor, filter, channel, x, y);
            }
        }
    }

    private void convoluteDeltaKernel(Tensor inTensor, Tensor outTensor, int filter, int channel, int x, int y) {
        int outIndex = filter * (kernelShape.dimension[0] * kernelShape.dimension[1] * inputShape.dimension[2])
                + channel * (kernelShape.dimension[0] * kernelShape.dimension[1])
                + y * (kernelShape.dimension[0]) + x;
        for (int i = 0; i < outputShape.dimension[0]; i++) {
            for (int j = 0; j < outputShape.dimension[1]; j++) {
                int pixelIndex = channel * (inputShape.dimension[0] * inputShape.dimension[1])
                        + (y + j) * inputShape.dimension[0] + (x + i);
                int deltaIndex = filter * (outputShape.dimension[0] * outputShape.dimension[1])
                        + j * outputShape.dimension[0] + i;
                weights.delta[outIndex] += inTensor.elements[pixelIndex] * outTensor.delta[deltaIndex];
            }
        }
    }

    @Override
    public String toString() {
        return "2D Convolutional Layer\n" +
                "\tKernel Size: " + kernelShape + " - " +
                "Stride: " + stride + "\n" +
                super.toString();
    }

    private Shape calculateOutputShape() {
        if (inputShape.dimension.length != 3) {
            throw new InputShapeException("Conv2D has wrong input shape, needed: (x, y, n_channels)");
        }
        if (kernelShape.dimension.length != 3) {
            throw new InputShapeException("Invalid kernel shape, needed: (x, y, n_filters)");
        }
        if (stride.volume > 1) {
            throw new java.lang.UnsupportedOperationException("Operation not yet implemented");
        }

        switch (padding) {
            case HALF:
                bPadX = padX = kernelShape.dimension[0] / 2;
                bPadY = padY = kernelShape.dimension[1] / 2;
                break;
            case FULL:
                bPadY = bPadX = 0;
                padX = kernelShape.dimension[0] - 1;
                padY = kernelShape.dimension[1] - 1;
                break;
            default:
            case NONE:
                bPadX = kernelShape.dimension[0] - 1;
                bPadY = kernelShape.dimension[1] - 1;
                padX = padY = 0;
        }

        int oX = (inputShape.dimension[0] + 2 * padX - kernelShape.dimension[0]) / stride.dimension[0] + 1;
        int oY = (inputShape.dimension[1] + 2 * padY - kernelShape.dimension[1]) / stride.dimension[1] + 1;

        return new Shape(oX, oY, kernelShape.dimension[2]);
    }
}