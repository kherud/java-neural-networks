package layers;

import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.tensor.initialiser.ZeroInitialiser;
import nn.layers.trainable.FullyConnectedLayer;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class FullyConnectedTest {

    private FullyConnectedLayer layer1;
    private FullyConnectedLayer layer2;

    @Before
    public void setup() {
        layer1 = new FullyConnectedLayer(new Shape(3), new Shape(3), new ZeroInitialiser());
        layer1.weights.elements = new float[]{-.5057f, .3356f, -.3485f, .3987f, .1673f, -.4597f, -.8943f, .8321f, -.1121f};

        layer2 = new FullyConnectedLayer(new Shape(3), new Shape(2), new ZeroInitialiser());
        layer2.weights.elements = new float[]{.4047f, -.8192f, .3662f, .9563f, -.1274f, -.7252f,};
    }


    @Test
    public void testForward1() {
        Tensor input = new Tensor(new Shape(3), new float[]{0.4183f, 0.5209f, 0.0291f});
        Tensor output = new Tensor(new Shape(3));

        layer1.forward(input, output);

        float[] expected = new float[]{-.0469f, .2406f, .0561f};
        for (int i = 0; i < expected.length; i++) {
            Assert.assertEquals(expected[i], output.elements[i], 0.0001);
        }
    }

    @Test
    public void testForward2() {
        Tensor input = new Tensor(new Shape(3), new float[]{.4883f, 0.5599f, 0.5140f});
        Tensor output = new Tensor(new Shape(2));

        layer2.forward(input, output);

        float[] expected = new float[]{-.0728f, .0229f};
        for (int i = 0; i < expected.length; i++) {
            Assert.assertEquals(expected[i], output.elements[i], 0.0001);
        }
    }

    @Test
    public void testBackward1() {
        Tensor input = new Tensor(new Shape(3), new float[3], new float[]{.0451f, .0557f, -.0891f});
        Tensor output = new Tensor(new Shape(3));

        layer1.backward(output, input);

        float[] expected = new float[]{0.07912705f, -0.04970998f, -0.03132404f};
        for (int i = 0; i < expected.length; i++) {
            Assert.assertEquals(expected[i], output.delta[i], 0.0001);
        }
    }

    @Test
    public void testBackward2() {
        Tensor input = new Tensor(new Shape(2), new float[2], new float[]{-0.3268f, 0.3268f});
        Tensor output = new Tensor(new Shape(3));

        layer2.backward(output, input);

        float[] expected = new float[]{0.1803f, 0.2261f, -0.3567f};
        for (int i = 0; i < expected.length; i++) {
            Assert.assertEquals(expected[i], output.delta[i], 0.0001);
        }
    }

    @Test
    public void calculateDeltaWeights1() {
        Tensor input = new Tensor(new Shape(3), new float[3], new float[]{0.0451f, 0.0557f, -0.0891f});
        Tensor output = new Tensor(new Shape(3), new float[]{0.4183f, 0.5209f, 0.0291f});

        layer1.calculateDeltaWeights(input, output);

        float[] expectedWeightDeltas = new float[]{.0188f, .0235f, .0013f, .0233f, .0290f, .0016f, -.0373f, -.0464f, -.0026f};
        for (int i = 0; i < expectedWeightDeltas.length; i++) {
            Assert.assertEquals(expectedWeightDeltas[i], layer1.weights.delta[i], 0.0001);
        }

        float[] expectedBiasDeltas = new float[]{.0451f, .0557f, -.0891f};
        for (int i = 0; i < expectedBiasDeltas.length; i++) {
            Assert.assertEquals(expectedBiasDeltas[i], layer1.bias.delta[i], 0.0001);
        }
    }

    @Test
    public void calculateDeltaWeights2() {
        Tensor input = new Tensor(new Shape(2), new float[2], new float[]{-0.3268f, 0.3268f});
        Tensor output = new Tensor(new Shape(3), new float[]{0.4883f, 0.5599f, 0.5140f});

        layer2.calculateDeltaWeights(input, output);

        float[] expectedWeightDeltas = new float[]{-.1596f, -.1830f, -.1680f, .1596f, .1830f, .1680f};
        for (int i = 0; i < expectedWeightDeltas.length; i++) {
            Assert.assertEquals(expectedWeightDeltas[i], layer2.weights.delta[i], 0.0001);
        }

        float[] expectedBiasDeltas = new float[]{-0.3268f, 0.3268f};
        for (int i = 0; i < expectedBiasDeltas.length; i++) {
            Assert.assertEquals(expectedBiasDeltas[i], layer2.bias.delta[i], 0.0001);
        }
    }
}
