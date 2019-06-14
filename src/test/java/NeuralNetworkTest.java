import nn.NeuralNetwork;
import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.tensor.initialiser.GaussianInitialiser;
import nn.layers.activation.ReLULayer;
import nn.layers.activation.SigmoidLayer;
import nn.layers.activation.SoftmaxLayer;
import nn.tensor.initialiser.XavierInitialiser;
import nn.tensor.initialiser.ZeroInitialiser;
import nn.layers.trainable.Conv2DLayer;
import nn.layers.trainable.FullyConnectedLayer;
import nn.layers.trainable.TrainableLayer;
import nn.layers.utility.Dropout;
import nn.layers.utility.MaxPooling2DLayer;
import nn.loss.CrossEntropy;
import nn.loss.Loss;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNetworkTest {

    private NeuralNetwork nn1;
    private NeuralNetwork nn2;
    private NeuralNetwork nn3;
    private NeuralNetwork nn4;
    private Tensor input, label;
    private Loss lossFunction;


    @Before
    public void setup() {
        nn1 = new NeuralNetwork(
                new FullyConnectedLayer(new Shape(3), new Shape(3), new ZeroInitialiser()),
                new SigmoidLayer(new Shape(3)),
                new FullyConnectedLayer(new Shape(3), new Shape(2), new ZeroInitialiser()),
                new SoftmaxLayer(new Shape(2))
        );

        ((TrainableLayer) nn1.layers[0]).weights.elements = new float[]{-.5057f, .3356f, -.3485f, .3987f, .1673f, -.4597f, -.8943f, .8321f, -.1121f};
        ((TrainableLayer) nn1.layers[2]).weights.elements = new float[]{.4047f, -.8192f, .3662f, .9563f, -.1274f, -.7252f};

        input = new Tensor(new Shape(3), new float[]{0.4183f, 0.5209f, 0.0291f});
        label = new Tensor(new Shape(2), new float[]{0.7095f, 0.0942f});

        lossFunction = new CrossEntropy();

        nn2 = new NeuralNetwork(
                new Conv2DLayer(new Shape(28, 28, 1), new Shape(3, 3, 24), new Shape(1, 1), Conv2DLayer.Padding.NONE, new XavierInitialiser()),
                new Conv2DLayer(new Shape(26, 26, 24), new Shape(3, 3, 8), new Shape(1, 1), Conv2DLayer.Padding.NONE, new XavierInitialiser()),
                new FullyConnectedLayer(new Shape(24 * 24 * 8), new Shape(300), new XavierInitialiser()),
                new SigmoidLayer(new Shape(300)),
                new FullyConnectedLayer(new Shape(300), new Shape(10), new XavierInitialiser()),
                new SoftmaxLayer(new Shape(10))
        );

        nn3 = new NeuralNetwork(
                new FullyConnectedLayer(new Shape(3), new Shape(3), new ZeroInitialiser()),
                new SigmoidLayer(new Shape(3)),
                new Dropout(new Shape(3), 0.5f, new Random(1234)),
                new FullyConnectedLayer(new Shape(3), new Shape(2), new ZeroInitialiser()),
                new SoftmaxLayer(new Shape(2))
        );

        ((TrainableLayer) nn3.layers[0]).weights.elements = new float[]{-.5057f, .3356f, -.3485f, .3987f, .1673f, -.4597f, -.8943f, .8321f, -.1121f};
        ((TrainableLayer) nn3.layers[3]).weights.elements = new float[]{.4047f, -.8192f, .3662f, .9563f, -.1274f, -.7252f};

        nn4 = new NeuralNetwork(
                new Conv2DLayer(new Shape(28, 28, 1), // input shape
                        new Shape(3, 3, 18), // filter size
                        new Shape(1, 1), // stride
                        Conv2DLayer.Padding.NONE, // padding
                        new GaussianInitialiser(),
                        false),
                new ReLULayer(new Shape(26 * 26 * 18)),
                new MaxPooling2DLayer(
                        new Shape(26, 26, 18), // input shape
                        new Shape(3, 3), // pooling shape
                        new Shape(3, 3) // stride
                ),
                new Conv2DLayer(new Shape(8, 8, 18),
                        new Shape(3, 3, 6),
                        new Shape(1, 1),
                        Conv2DLayer.Padding.NONE,
                        new GaussianInitialiser()),
                new ReLULayer(new Shape(6 * 6 * 6)),
                new FullyConnectedLayer(new Shape(6 * 6 * 6),
                        new Shape(10),
                        new XavierInitialiser()),
                new SoftmaxLayer(new Shape(10))
        );
    }

    @Test
    public void testForward1() {
        nn1.forward(input);

        List<float[]> expected = new ArrayList<>();
        expected.add(new float[]{-0.0469f, 0.2406f, 0.0561f});
        expected.add(new float[]{0.4883f, 0.5599f, 0.5140f});
        expected.add(new float[]{-0.0728f, 0.0229f});
        expected.add(new float[]{0.4761f, 0.5239f});

        Assert.assertEquals(expected.size(), nn1.tensors.length);

        for (int i = 0; i < expected.size(); i++) {
            Assert.assertEquals(expected.get(i).length, nn1.tensors[i].elements.length);

            for (int j = 0; j < nn1.tensors[i].elements.length; j++) {
                Assert.assertEquals(expected.get(i)[j], nn1.tensors[i].elements[j], 0.0001);
            }
        }
    }

    @Test
    public void testForward3TRAIN() {
        nn3.state = NeuralNetwork.State.TRAIN;
        nn3.forward(input);

        List<float[]> expected = new ArrayList<>();
        expected.add(new float[]{-0.0469f, 0.2406f, 0.0561f});
        expected.add(new float[]{0.4883f, 0.5599f, 0.5140f});
        expected.add(new float[]{0.4883f, 0, 0.5140f});
        expected.add(new float[]{0.38584362f, 0.0941816f});
        expected.add(new float[]{0.57240297f, 0.42759703f});

        Assert.assertEquals(expected.size(), nn3.tensors.length);

        for (int i = 0; i < expected.size(); i++) {
            Assert.assertEquals(expected.get(i).length, nn3.tensors[i].elements.length);

            for (int j = 0; j < nn3.tensors[i].elements.length; j++) {
                Assert.assertEquals(expected.get(i)[j], nn3.tensors[i].elements[j], 0.0001);
            }
        }
    }

    @Test
    public void testForward3PREDICT(){
        nn3.state = NeuralNetwork.State.PREDICT;
        nn3.forward(input);

        List<float[]> expected = new ArrayList<>();
        expected.add(new float[]{-0.0469f, 0.2406f, 0.0561f});
        expected.add(new float[]{0.4883f, 0.5599f, 0.5140f});
        expected.add(new float[]{0, 0, 0});
        expected.add(new float[]{-0.0728f, 0.0229f});
        expected.add(new float[]{0.4761f, 0.5239f});

        Assert.assertEquals(expected.size(), nn3.tensors.length);

        for (int i = 0; i < expected.size(); i++) {
            Assert.assertEquals(expected.get(i).length, nn3.tensors[i].elements.length);

            for (int j = 0; j < nn3.tensors[i].elements.length; j++) {
                Assert.assertEquals(expected.get(i)[j], nn3.tensors[i].elements[j], 0.0001);
            }
        }
    }

    @Test
    public void testNN2(){
        Tensor input = new Tensor(new Shape(28, 28, 1), new float[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,18,18,18,126,136,175,26,166,255,247,127,0,0,0,0,0,0,0,0,0,0,0,0,30,36,94,154,170,253,253,253,253,253,225,172,253,242,195,64,0,0,0,0,0,0,0,0,0,0,0,49,238,253,253,253,253,253,253,253,253,251,93,82,82,56,39,0,0,0,0,0,0,0,0,0,0,0,0,18,219,253,253,253,253,253,198,182,247,241,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,80,156,107,253,253,205,11,0,43,154,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,1,154,253,90,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,139,253,190,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,190,253,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,241,225,160,108,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,81,240,253,253,119,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,186,253,253,150,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,93,252,253,187,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,249,253,249,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,46,130,183,253,253,207,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,39,148,229,253,253,253,250,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,114,221,253,253,253,253,201,78,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,66,213,253,253,253,253,198,81,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,171,219,253,253,253,253,195,80,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,172,226,253,253,253,253,244,133,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,136,253,253,253,212,135,132,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0});
        Tensor label = new Tensor(new Shape(10), new float[]{0,0,0,0,0,1,0,0,0,0});

        nn2.forward(input);
        lossFunction.backward(nn2.tensors[nn2.tensors.length-1], label);
        nn2.backward(input);
    }

    @Test
    public void testNN4(){
        Tensor input = new Tensor(new Shape(28, 28, 1), new float[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,18,18,18,126,136,175,26,166,255,247,127,0,0,0,0,0,0,0,0,0,0,0,0,30,36,94,154,170,253,253,253,253,253,225,172,253,242,195,64,0,0,0,0,0,0,0,0,0,0,0,49,238,253,253,253,253,253,253,253,253,251,93,82,82,56,39,0,0,0,0,0,0,0,0,0,0,0,0,18,219,253,253,253,253,253,198,182,247,241,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,80,156,107,253,253,205,11,0,43,154,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,1,154,253,90,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,139,253,190,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,190,253,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,241,225,160,108,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,81,240,253,253,119,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,186,253,253,150,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,93,252,253,187,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,249,253,249,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,46,130,183,253,253,207,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,39,148,229,253,253,253,250,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,114,221,253,253,253,253,201,78,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,66,213,253,253,253,253,198,81,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,171,219,253,253,253,253,195,80,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,172,226,253,253,253,253,244,133,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,136,253,253,253,212,135,132,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0});
        Tensor label = new Tensor(new Shape(10), new float[]{0,0,0,0,0,1,0,0,0,0});

        nn4.forward(input);
        lossFunction.backward(nn4.tensors[nn4.tensors.length-1], label);
        nn4.backward(input);

        /*for (Layer layer : nn4.layers){
            if (layer instanceof TrainableLayer){
                Tensor weights = ((TrainableLayer) layer).weights;
                for (int i = 0; i < weights.elements.length; i++)
                    System.out.print(weights.elements[i] + ", ");
            }
            System.out.println();
        }

        System.out.println("deltas");

        for (Layer layer : nn4.layers){
            if (layer instanceof TrainableLayer){
                Tensor weights = ((TrainableLayer) layer).weights;
                for (int i = 0; i < weights.delta.length; i++)
                System.out.print(weights.delta[i] + ", ");
            }
            System.out.println();
        }*/
    }

    @Test
    public void testBackward1() {
        nn1.forward(input);
        lossFunction.backward(nn1.tensors[nn1.tensors.length-1], label);
        nn1.backward(input);

        List<float[]> expectedDeltas = new ArrayList<>();
        expectedDeltas.add(new float[]{0.0451f, 0.0557f, -0.0891f});
        expectedDeltas.add(new float[]{-0.3268f, 0.3268f});

        List<float[]> expectedWeightDeltas = new ArrayList<>();
        expectedWeightDeltas.add(new float[]{.0188f, .0235f, .0013f, .0233f, .0290f, .0016f, -.0373f, -.0464f, -.0026f});
        expectedWeightDeltas.add(new float[]{-.1596f, -.1830f, -.1680f, .1596f, .1830f, .1680f});

        Assert.assertEquals(expectedWeightDeltas.get(0).length, ((TrainableLayer) nn1.layers[0]).weights.delta.length);
        for (int i = 0; i < ((TrainableLayer) nn1.layers[0]).weights.delta.length; i++){
            Assert.assertEquals(expectedWeightDeltas.get(0)[i], ((TrainableLayer) nn1.layers[0]).weights.delta[i], 0.0001);
        }

        Assert.assertEquals(expectedDeltas.get(0).length, nn1.tensors[0].delta.length);
        for (int i = 0; i < nn1.tensors[0].delta.length; i++){
            Assert.assertEquals(expectedDeltas.get(0)[i], nn1.tensors[0].delta[i], 0.0001);
        }

        Assert.assertEquals(expectedWeightDeltas.get(1).length, ((TrainableLayer) nn1.layers[2]).weights.delta.length);
        for (int i = 0; i < ((TrainableLayer) nn1.layers[2]).weights.delta.length; i++){
            Assert.assertEquals(expectedWeightDeltas.get(1)[i], ((TrainableLayer) nn1.layers[2]).weights.delta[i], 0.0001);
        }

        Assert.assertEquals(expectedDeltas.get(1).length, nn1.tensors[2].delta.length);
        for (int i = 0; i < nn1.tensors[2].delta.length; i++){
            Assert.assertEquals(expectedDeltas.get(1)[i], nn1.tensors[2].delta[i], 0.0001);
        }
    }

    @Test
    public void testBackward3TRAIN(){
        nn3.state = NeuralNetwork.State.TRAIN;
        nn3.forward(input);
        lossFunction.backward(nn3.tensors[nn3.tensors.length-1], new Tensor(new Shape(2), new float[]{0.7095f, 0.0942f}));
        nn3.backward(input);

        List<float[]> expectedDeltas = new ArrayList<>();
        expectedDeltas.add(new float[]{0.03438161f,  0, -0.06801157f});
        expectedDeltas.add(new float[]{-0.24945975f,  0.24945971f});

        List<float[]> expectedWeightDeltas = new ArrayList<>();
        expectedWeightDeltas.add(new float[]{0.01438183f,  0.01790938f,  0.0010005f, 0, 0, 0, -0.02844924f, -0.03542723f, -0.00197914f});
        expectedWeightDeltas.add(new float[]{-0.12180789f,  0, -0.1282272f, 0.12180787f, 0, 0.12822718f});

        Assert.assertEquals(expectedWeightDeltas.get(0).length, ((TrainableLayer) nn3.layers[0]).weights.delta.length);
        for (int i = 0; i < ((TrainableLayer) nn3.layers[0]).weights.delta.length; i++){
            Assert.assertEquals(expectedWeightDeltas.get(0)[i], ((TrainableLayer) nn3.layers[0]).weights.delta[i], 0.0001);
        }

        Assert.assertEquals(expectedDeltas.get(0).length, nn3.tensors[0].delta.length);
        for (int i = 0; i < nn3.tensors[0].delta.length; i++){
            Assert.assertEquals(expectedDeltas.get(0)[i], nn3.tensors[0].delta[i], 0.0001);
        }

        Assert.assertEquals(expectedWeightDeltas.get(1).length, ((TrainableLayer) nn3.layers[3]).weights.delta.length);
        for (int i = 0; i < ((TrainableLayer) nn3.layers[3]).weights.delta.length; i++){
            Assert.assertEquals(expectedWeightDeltas.get(1)[i], ((TrainableLayer) nn3.layers[3]).weights.delta[i], 0.0001);
        }

        Assert.assertEquals(expectedDeltas.get(1).length, nn3.tensors[3].delta.length);
        for (int i = 0; i < nn3.tensors[3].delta.length; i++){
            Assert.assertEquals(expectedDeltas.get(1)[i], nn3.tensors[3].delta[i], 0.0001);
        }
    }

    @Test(expected = IllegalStateException.class)
    public void testBackward3PREDICT(){
        nn3.state = NeuralNetwork.State.PREDICT;
        nn3.forward(input);
        lossFunction.backward(nn3.tensors[nn3.tensors.length-1], new Tensor(new Shape(2), new float[]{0.7095f, 0.0942f}));
        nn3.backward(input);
    }

}
