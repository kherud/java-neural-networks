package layers;

import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.layers.activation.ActivationLayer;
import nn.layers.activation.SigmoidLayer;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class SigmoidTest {

    ActivationLayer layer;

    @Before
    public void setup(){
       layer = new SigmoidLayer(new Shape(3));
    }

    @Test
    public void testForward(){
        Tensor input = new Tensor(new Shape(3), new float[]{-0.0469f, 0.2406f, 0.0561f});
        Tensor output = new Tensor(new Shape(3));

        layer.forward(input, output);

        float[] expected = new float[]{0.4883f, 0.5599f, 0.5140f};
        for (int i = 0; i < expected.length; i++){
            Assert.assertEquals(expected[i], output.elements[i], 0.0001);
        }
    }

    @Test
    public void testBackward(){
        Tensor input = new Tensor(new Shape(3), new float[]{0.4883f, 0.5599f, 0.5140f}, new float[]{0.1803f, 0.2261f, -0.3567f});
        Tensor output = new Tensor(new Shape(3)); // output of forward pass

        layer.backward(output, input);

        float[] expected = new float[]{0.0451f, 0.0557f, -0.0891f};
        for (int i = 0; i < expected.length; i++){
            Assert.assertEquals(expected[i], output.delta[i],0.0001);
        }
    }
}
