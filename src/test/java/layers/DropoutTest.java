package layers;

import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.layers.utility.Dropout;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

public class DropoutTest {

    private Dropout layer1;
    private Dropout layer2;
    private final Random random;

    // first ten values are
    // >> 0,647, 0,260, 0,951, 0,237, 0,858, 0,317, 0,458, 0,550, 0,336, 0,093,
    public DropoutTest(){
        random = new Random(1234);
    }

    @Before
    public void setup(){
        Shape inputShape = new Shape(10);
        this.layer1 = new Dropout(inputShape, 0.4f, random);

        Shape inputShape2 = new Shape(2, 2, 2);
        this.layer2 = new Dropout(inputShape2, 0.6f, random);
    }

    @Test
    public void testForward1(){
        Tensor input = new Tensor(layer1.inputShape, new float[]{4, 3, -6, 1, -7, -7, 1, -2, -8, -5});
        Tensor output = new Tensor(layer1.outputShape);

        // >> 0,647, 0,260, 0,951, 0,237, 0,858, 0,317, 0,458, 0,550, 0,336, 0,093,
        layer1.forward(input, output);

        float[] expectedOutput = new float[]{4, 0, -6, 0, -7, 0, 1, -2, 0, 0};

        for (int i = 0; i < expectedOutput.length; i++){
            Assert.assertEquals(expectedOutput[i], output.elements[i], 0.0001f);
        }
    }

    @Test
    public void testForward2(){
        Tensor input = new Tensor(layer2.inputShape, new float[]{-8, -3, -2,  7,  3,  0,  4, -4});
        Tensor output = new Tensor(layer2.outputShape);

        // >> 0,647, 0,260, 0,951, 0,237, 0,858, 0,317, 0,458, 0,550, 0,336, 0,093,
        layer2.forward(input, output);

        float[] expectedOutput = new float[]{-8, 0, -2, 0, 3, 0, 0, 0};

        for (int i = 0; i < expectedOutput.length; i++){
            Assert.assertEquals(expectedOutput[i], output.elements[i], 0.0001f);
        }
    }

    @Test
    public void testBackward1(){
        Tensor input = new Tensor(layer1.outputShape, null, new float[]{4, 3, -6, 1, -7, -7, 1, -2, -8, -5});
        Tensor output = new Tensor(layer1.inputShape);

        layer1.mask = new float[]{4, 0, -6, 0, -7, 0, 1, -2, 0, 0};
        layer1.backward(output, input);

        float[] expectedOutput = new float[]{4, 0, -6, 0, -7, 0, 1, -2, 0, 0};

        for (int i = 0; i < expectedOutput.length; i++){
            Assert.assertEquals(expectedOutput[i], output.delta[i], 0.0001f);
        }
    }

    @Test
    public void testBackward2(){
        Tensor input = new Tensor(layer2.outputShape, null, new float[]{-8, 1, -2, 1, 3, 1, 1, 1});
        Tensor output = new Tensor(layer2.inputShape);

        layer2.mask = new float[]{-8, 0, -2, 0, 3, 0, 0, 0};
        layer2.backward(output, input);

        float[] expectedOutput = new float[]{-8, 0, -2, 0, 3, 0, 0, 0};

        for (int i = 0; i < expectedOutput.length; i++){
            Assert.assertEquals(expectedOutput[i], output.delta[i], 0.0001f);
        }
    }

}
