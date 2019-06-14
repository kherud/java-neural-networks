package nn.tensor;

import java.util.Arrays;

public class Shape {
    public final int[] dimension;
    public final int volume;

    public Shape(int... dimensions) {
        dimension = new int[dimensions.length];
        System.arraycopy(dimensions, 0, dimension, 0, dimensions.length);

        int volume = 1;
        for (int dimension : this.dimension) {
            volume *= dimension;
        }
        this.volume = volume;
    }

    public Shape(Shape... shapes) {
        int depth = 0;
        for (Shape shape : shapes) {
            depth += shape.dimension.length;
        }
        dimension = new int[depth];
        int index = 0;
        for (Shape shape : shapes) {
            System.arraycopy(shape.dimension, 0, dimension, index, shape.dimension.length);
            index += shape.dimension.length;
        }

        int volume = 1;
        for (int dimension : this.dimension) {
            volume *= dimension;
        }
        this.volume = volume;
    }

    @Override
    public String toString(){
        return Arrays.toString(dimension);
    }

    @Override
    public boolean equals(Object o) {
        if (o == this) {
            return true;
        }

        if (!(o instanceof Shape)) {
            return false;
        }

        Shape s = (Shape) o;

        for (int i = 0; i < s.dimension.length; i++) {
            if (this.dimension[i] != s.dimension[i]) return false;
        }

        return true;
    }

}
