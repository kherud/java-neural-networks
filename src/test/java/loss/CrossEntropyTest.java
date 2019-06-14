package loss;

import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.loss.CrossEntropy;
import nn.loss.Loss;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class CrossEntropyTest {

    private Loss lossFunction;

    @Before
    public void setup(){
        this.lossFunction = new CrossEntropy();
    }

    @Test
    public void testForward(){
        Tensor prediction = new Tensor(new Shape(2), new float[]{0.4761f, 0.5239f});
        Tensor truth = new Tensor(new Shape(2), new float[]{0.7095f, 0.0942f});

        lossFunction.forward(prediction, truth);

        float[] expected = new float[]{0.5874280354780502f};
        Assert.assertEquals(expected.length, lossFunction.loss.shape.volume);
        Assert.assertEquals(expected[0], lossFunction.loss.elements[0], 0.0001);
    }

    @Test
    public void testBackward(){
        Tensor prediction = new Tensor(new Shape(2), new float[]{0.4761f, 0.5239f});
        Tensor truth = new Tensor(new Shape(2), new float[]{0.7095f, 0.0942f});

        lossFunction.backward(prediction, truth);

        float[] expected = new float[]{-1.4902f, -0.1798f};
        Assert.assertEquals(expected.length, prediction.delta.length);
        for (int i = 0; i < expected.length; i++) {
            Assert.assertEquals(expected[i], prediction.delta[i], 0.0001);
        }
    }
}
