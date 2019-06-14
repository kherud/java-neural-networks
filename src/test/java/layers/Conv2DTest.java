package layers;

import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.tensor.initialiser.ZeroInitialiser;
import nn.layers.trainable.Conv2DLayer;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class Conv2DTest {

    private Conv2DLayer layer1;
    private Conv2DLayer layer2;
    private Conv2DLayer layer3;

    @Before
    public void setup() {
        Shape inputShape1 = new Shape(4, 3, 2);
        Shape kernelShape1 = new Shape(2, 2, 2);
        Shape strideShape1 = new Shape(1, 1);
        layer1 = new Conv2DLayer(inputShape1, kernelShape1, strideShape1, Conv2DLayer.Padding.NONE, new ZeroInitialiser());
        layer1.weights.elements = new float[]{0.1f, -0.2f, 0.3f, 0.4f, 0.7f, 0.6f, 0.9f, -1.1f, 0.37f, -0.9f, 0.32f, 0.17f, 0.9f, 0.3f, 0.2f, -0.7f};

        Shape inputShape2 = new Shape(28, 28, 1);
        Shape kernelShape2 = new Shape(3, 3, 16);
        Shape strideShape2 = new Shape(1, 1);
        layer2 = new Conv2DLayer(inputShape2, kernelShape2, strideShape2, Conv2DLayer.Padding.FULL, new ZeroInitialiser());

        Shape inputShape3 = new Shape(3, 4, 3);
        Shape kernelShape3 = new Shape(2, 3, 2);
        Shape strideShape3 = new Shape(1, 1);
        layer3 = new Conv2DLayer(inputShape3, kernelShape3, strideShape3, Conv2DLayer.Padding.NONE, new ZeroInitialiser());
        layer3.weights.elements = new float[]{2, -3, -7, -3,  5,  2, -3, -2,  8, -9,  6, -9,  1, -5,  0,  1,  2, -4,  1,  7, -6, -1, -5, -6, -3, -4,  8, -4,  8, -3,  5, -6, -7,  7, -6, -7};;
    }

    @Test
    public void testWeightDeltaIndex(){
        Shape inputShape1 = new Shape(4, 4, 2);
        Shape kernelShape1 = new Shape(2, 2, 2);
        Shape strideShape1 = new Shape(1, 1);
        Conv2DLayer layer = new Conv2DLayer(inputShape1, kernelShape1, strideShape1, Conv2DLayer.Padding.NONE, new ZeroInitialiser());

        Tensor input = new Tensor(layer.inputShape, new float[]{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

        Tensor deltas = new Tensor(layer.outputShape, new float[layer.outputShape.volume], new float[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2});

        layer.calculateDeltaWeights(deltas, input);

        float[] expected = new float[]{18, 18, 18, 18, 9, 9, 9, 9, 36, 36, 36, 36, 18, 18, 18, 18};
        for (int i = 0; i < layer.weights.delta.length; i++){
            Assert.assertEquals(expected[i], layer.weights.delta[i], 0.0001f);
        }

        layer.weights.elements = new float[]{1, 1, 1, 1, 2, 2, 2, 2, 10, 10, 10, 10, 20, 20, 20, 20};

        layer.backward(input, deltas);

        float[] expectedDeltas = new float[]{21, 42, 42, 21, 42, 84, 84, 42, 42, 84, 84, 42, 21, 42, 42, 21, 42, 84, 84, 42, 84, 168, 168, 84, 84, 168, 168, 84, 42, 84, 84, 42};
        for (int i = 0; i < input.delta.length; i++){
            Assert.assertEquals(expectedDeltas[i], input.delta[i], 0.0001f);
        }
    }

    @Test
    public void testForward1() {
        Tensor input = new Tensor(layer1.inputShape, new float[]{0.1f, -0.2f, 0.5f, 0.6f, 1.2f, 1.4f, 1.6f, 2.2f, 0.01f, 0.2f, -0.3f, 4.0f, 0.9f, 0.3f, 0.5f, 0.65f, 1.1f, 0.7f, 2.2f, 4.4f, 3.2f, 1.7f, 6.3f, 8.2f});
        Tensor output = new Tensor(layer1.outputShape);

        layer1.forward(input, output);

        float[] expected = new float[]{2.0f, -0.34000015f, -0.8299999f, 2.123f, -3.8300004f, 2.0599995f, 1.469f, -0.7839999f, -1.4639999f, -0.12880003f, -3.6889997f, -1.9839993f};
        for (int i = 0; i < expected.length; i++) {
            Assert.assertEquals(expected[i], output.elements[i], 0.0001);
        }
    }

    @Test
    public void testForward3(){
        Tensor input = new Tensor(layer3.inputShape, new float[]{-9, -3, -5,  0,  7,  4, -6, -3,  1,  5,  2, -2,  3,  8,  8, -3,  1, 2, -5, -6, -5,  8,  0, -8, -8,  5, -6,  2, -5,  7, -5,  6, -4,  0, 8,  2});
        Tensor output = new Tensor(layer3.outputShape);

        layer3.forward(input, output);

        float[] expected = new float[]{-172, -36, 129, 52, -211, -27, 165, -160};
        for (int i = 0; i < expected.length; i++) {
            Assert.assertEquals(expected[i], output.elements[i], 0.0001);
        }
    }

    @Test
    public void testBackward1() {
        Tensor input = new Tensor(layer1.outputShape, new float[layer1.outputShape.volume], new float[]{0.1f, 0.33f, -0.6f, -0.25f, 1.3f, 0.01f, -0.5f, 0.2f, 0.1f, -0.8f, 0.81f, 1.1f});
        Tensor output = new Tensor(layer1.inputShape);

        layer1.backward(output, input);

        float[] expected = new float[]{-0.175f, 0.537f, -0.269f, 0.030000009f, -0.451f, 1.3177f, -0.5629999f, -1.215f, -0.33100003f, 0.41320002f, 1.0127001f, 0.191f, -0.38f, 0.32099998f, -0.072000004f, -0.33f, -0.905f, 1.8259999f, 0.997f, 0.926f, -0.385f, 2.1669998f, -1.7679999f, -0.78099996f};
        for (int i = 0; i < expected.length; i++) {
            Assert.assertEquals(expected[i], output.delta[i], 0.0001);
        }
    }

    @Test
    public void testBackward3(){
        Tensor input = new Tensor(layer3.outputShape, new float[layer3.outputShape.volume], new float[]{-4,  3, -4, -6, -8, -8,  8,  5});
        Tensor output = new Tensor(layer3.inputShape);

        layer3.backward(output, input);

        float[] expected = new float[]{-16, -46, -65, 76, 108, 52, 0, 111, 67, -60, -111, -42, 36, 55, 26, -108, 7, -3, -56, 10, 31, 40, 16, 39, -44, 31, 33, 92, -13, -53, -16, 143, 73, -56, -82, -11};
        for (int i = 0; i < expected.length; i++) {
            Assert.assertEquals(expected[i], output.delta[i], 0.0001);
        }
    }

    @Test
    public void testCalculateDeltaWeights1() {
        Tensor input = new Tensor(layer1.inputShape, new float[]{0.1f, -0.2f, 0.5f, 0.6f, 1.2f, 1.4f, 1.6f, 2.2f, 0.01f, 0.2f, -0.3f, 4.0f, 0.9f, 0.3f, 0.5f, 0.65f, 1.1f, 0.7f, 2.2f, 4.4f, 3.2f, 1.7f, 6.3f, 8.2f});
        Tensor output = new Tensor(layer1.outputShape, new float[layer1.outputShape.volume], new float[]{0.1f, 0.33f, -0.6f, -0.25f, 1.3f, 0.01f, -0.5f, 0.2f, 0.1f, -0.8f, 0.81f, 1.1f});

        layer1.calculateDeltaWeights(output, input);

        float[] expectedWeighDeltas = new float[]{1.18f, 1.5369998f, -0.12350003f, -1.052f, 0.54599994f, 2.5339997f, 0.494f, 6.0029993f, 1.894f, 2.856f, -0.33600003f, 3.8370001f, 1.767f, 6.077f, 5.557f, 13.293001f};
        for (int i = 0; i < expectedWeighDeltas.length; i++) {
            Assert.assertEquals(expectedWeighDeltas[i], layer1.weights.delta[i], 0.0001);
        }

        float[] expectedBiasDeltas = new float[]{0.89f, 0.91f};
        for (int i = 0; i < expectedBiasDeltas.length; i++){
            Assert.assertEquals(expectedBiasDeltas[i], layer1.bias.delta[i], 0.0001);
        }
    }

    @Test
    public void testCalculateDeltaWeights3(){
        Tensor input = new Tensor(layer3.inputShape, new float[]{-9, -3, -5,  0,  7,  4, -6, -3,  1,  5,  2, -2,  3,  8,  8, -3,  1, 2, -5, -6, -5,  8,  0, -8, -8,  5, -6,  2, -5,  7, -5,  6, -4,  0, 8,  2});
        Tensor output = new Tensor(layer3.outputShape, new float[layer3.outputShape.volume], new float[]{-4,  3, -4, -6, -8, -8,  8,  5});

        layer3.calculateDeltaWeights(output, input);

        float[] expectedWeighDeltas = new float[]{-15, -55, 63, -10, -17, 19, 18, -24, 71, 56, -30, 57, 69, -60, -39, 41, -10, -80, 131, 140, -119, -107, 122, 22, -107, -110, -54, -97, 152, 48, 15, 3, 14, 12, 32, 58};
        for (int i = 0; i < expectedWeighDeltas.length; i++) {

            Assert.assertEquals(expectedWeighDeltas[i], layer3.weights.delta[i], 0.0001);
        }

        float[] expectedBiasDeltas = new float[]{-11, -3};
        for (int i = 0; i < expectedBiasDeltas.length; i++){
            Assert.assertEquals(expectedBiasDeltas[i], layer3.bias.delta[i], 0.0001);
        }
    }

    @Test
    public void testCalculateOutputShape1() {
        Shape expectedShape = new Shape(3, 2, 2);
        Assert.assertEquals(expectedShape, layer1.outputShape);
    }

    @Test
    public void testCalculateOutputShape2() {
        Shape expectedShape = new Shape(30, 30, 16);
        Assert.assertEquals(expectedShape, layer2.outputShape);
    }

    @Test
    public void testCalculateOutputShape3() {
        Shape expectedShape = new Shape(2, 2, 2);
        Assert.assertEquals(expectedShape, layer3.outputShape);
    }


}
