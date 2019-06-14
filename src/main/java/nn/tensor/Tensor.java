package nn.tensor;

import nn.error.ShapeMismatchException;

public class Tensor {

    public float[] elements;
    public float[] delta;
    public Shape shape;

    public Tensor(Shape shape){
        this.shape = shape;
        elements = new float[shape.volume];
        delta = new float[shape.volume];
    }

    public Tensor(Shape shape, float[] elements){
        this.shape = shape;
        this.elements = elements;
        delta = new float[shape.volume];
    }

    public Tensor(Shape shape, float[] elements, float[] deltas){
        this.shape = shape;
        this.elements = elements;
        delta = deltas;
    }

    public void resetElements(){
        for (int i = 0; i < shape.volume; i++){
            elements[i] = 0;
        }
    }

    public void resetDeltas(){
        for (int i = 0; i < shape.volume; i++){
            delta[i] = 0;
        }
    }

    public Shape getShape(){
        return shape;
    }

    public void reshape(Shape shape){
        if (this.shape.volume != shape.volume)
            throw new ShapeMismatchException("Invalid reshape operation");
        this.shape = shape;
    }

    public float[] getElements(){
        return elements;
    }
}
