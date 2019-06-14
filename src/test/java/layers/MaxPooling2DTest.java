package layers;

import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.layers.Layer;
import nn.layers.utility.MaxPooling2DLayer;
import nn.layers.utility.PoolingLayer;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class MaxPooling2DTest {

    private Layer layer1;
    private Layer layer2;

    @Before
    public void setup(){
        this.layer1 = new MaxPooling2DLayer(
                new Shape(3, 2, 2),
                new Shape(2, 2),
                new Shape(1, 1)
        );

        this.layer2 = new MaxPooling2DLayer(
                new Shape(9, 9, 2),
                new Shape(3, 3),
                new Shape(3, 3)
        );
    }

    @Test
    public void testForward1(){
        Tensor input = new Tensor(layer1.inputShape, new float[]{2.0f, -0.34000015f, -0.8299999f, 2.123f, -3.8300004f, 2.0599995f, 1.469f, -0.7839999f, -1.4639999f, -0.12880003f, -3.6889997f, -1.9839993f});
        Tensor output = new Tensor(layer1.outputShape);

        layer1.forward(input, output);

        float[] expectedOutput = new float[]{2.123f, 2.0599995f, 1.469f, -0.7839999f};
        for (int i = 0; i < expectedOutput.length; i++){
            Assert.assertEquals(expectedOutput[i], output.elements[i], 0.0001f);
        }

        int[] expectedPoolingMask = new int[]{-1, -1, -1, 0, -1, 1, 2, 3, -1, -1, -1, -1};
        for (int i = 0; i < ((PoolingLayer) layer1).poolingMask.length; i++) {
            Assert.assertEquals(expectedPoolingMask[i], ((PoolingLayer) layer1).poolingMask[i]);
        }
    }

    @Test
    public void testForward2(){
        Tensor input = new Tensor(layer2.inputShape, new float[]{-5, 8, 3, -8, 1, -1, -4, 3, -1, 5, 3,-2, 8, 8, -9, 4, -4, 3, -8, 9, 3, -2, -8, 0, 1, 5, -6, 5, -8, -5, 8, -6, -3, 3, -6, 6, 8, -8, 8, -1, -9, 5, 1, 5, 6, 5, -1, -4, 3, 4, 8, -1, -1, 2, -8, 9, 0, -8, 8, -4, -8, 8, -1, 5, -8, 0, 9, -8, -3, 2, -6, 3, -7, -7, -8, 1, 9, -1, 7, 8, -1, -7, 7, 7, -1, 1, -2, 2, 9, 4, -1, -3, -6, 7, -4, 8, -5, 8, -8, 5, 6, 2, 6, 0, 3, -2, -5, -7, 5, -3, 5, -7, 8, -3, 1, -6, -2, 2, 0, 6, 3, -2, -4, -6, 8, -2, -2, -5, -1, 4, 8, -8, -7, -3, -6, -4, -1, -2, 9, -7, 9, -1, 8, -6, 5, 9, 1, -4, 1, -5, -3, 2, -6, 4, 9, 2, 6, -3, 1, -6, 7, 8});
        Tensor output = new Tensor(layer2.outputShape);

        layer2.forward(input, output);

        float[] expectedOutput = new float[]{9, 8, 5, 8, 8, 6, 9, 9, 8, 7, 8, 9, 6, 8, 8, 9, 9, 8};
        for (int i = 0; i < output.elements.length; i++){
            Assert.assertEquals(expectedOutput[i], output.elements[i], 0.0001f);
        }

        int[] expectedPoolingMask = new int[]{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, 4, -1, -1, -1, -1, 5, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, -1, -1, -1, -1, -1, 11, -1, -1, -1, -1, -1, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, -1, -1, -1, -1, -1, -1, 12, -1, -1, -1, -1, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 16, -1, -1, -1, 17, -1, -1, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        for (int i = 0; i < ((PoolingLayer) layer2).poolingMask.length; i++) {
            Assert.assertEquals(expectedPoolingMask[i], ((PoolingLayer) layer2).poolingMask[i]);
        }
    }

    @Test
    public void testBackward1(){
        ((PoolingLayer) layer1).poolingMask = new int[]{-1, -1, -1, 0, -1, 1, 2, 3, -1, -1, -1, -1};
        Tensor input = new Tensor(layer1.outputShape, null, new float[]{1, 1, 1, 1});
        Tensor output = new Tensor(layer1.inputShape);

        layer1.backward(output, input);

        float[] expectedDeltas = new float[]{0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0};
        for (int i = 0; i < expectedDeltas.length; i++){
            Assert.assertEquals(expectedDeltas[i], output.delta[i], 0.0001f);
        }
    }

    @Test
    public void testCalculateOutputShape1(){
        Shape expectedShape = new Shape(2, 1, 2);
        Assert.assertEquals(expectedShape, layer1.outputShape);
    }

    @Test
    public void testCalculateOutputShape2(){
        Shape expectedShape = new Shape(3, 3, 2);
        Assert.assertEquals(expectedShape, layer2.outputShape);
    }
}
