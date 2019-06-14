package layers;

import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.layers.activation.ActivationLayer;
import nn.layers.activation.SoftmaxLayer;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class SoftmaxTest {

    ActivationLayer layer;

    @Before
    public void setup(){
        layer = new SoftmaxLayer(new Shape(2));
    }

    @Test
    public void testForward(){
        Tensor input = new Tensor(new Shape(2), new float[]{-0.0728f, 0.0229f});
        Tensor output = new Tensor(new Shape(2));

        layer.forward(input, output);

        float[] expected = new float[]{0.4761f, 0.5239f};
        for (int i = 0; i < expected.length; i++){
            Assert.assertEquals(expected[i], output.elements[i], 0.0001);
        }
    }

    @Test
    public void testBackward(){
        Tensor input = new Tensor(new Shape(2), new float[]{0.4761f, 0.5239f}, new float[]{-1.4901f, -0.1798f});
        Tensor output = new Tensor(new Shape(2)); // // output of forward pass

        layer.backward(output, input);

        float[] expected = new float[]{-0.3268f, 0.3268f};
        for (int i = 0; i < expected.length; i++){
            Assert.assertEquals(expected[i], output.delta[i],0.0001);
        }
    }
}
