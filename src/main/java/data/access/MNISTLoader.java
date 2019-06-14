package data.access;

import nn.tensor.Shape;
import nn.tensor.Tensor;
import nn.error.ShapeMismatchException;

import java.io.File;
import java.io.FileNotFoundException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class MNISTLoader {

    private final URL CSV_FILE_TRAIN = ClassLoader.getSystemClassLoader().getResource("data/mnist_train.csv");
    private final URL CSV_FILE_TEST = ClassLoader.getSystemClassLoader().getResource("data/mnist_train.csv");
    private Scanner scanner;

    public Tensor[][] loadTrain(Shape shape) {
        if (CSV_FILE_TRAIN == null)
            throw new RuntimeException("Train CSV file cannot be found");
        if (shape.volume != 784)
            throw new ShapeMismatchException("Invalid shape for MNIST data");

        System.out.println("Loading train data...");
        Tensor[][] trainData = new Tensor[][]{new Tensor[60000], new Tensor[60000]};
        return loadCsvFile(new File(CSV_FILE_TRAIN.getFile()), trainData, shape);
    }

    public Tensor[][] loadTest(Shape shape){
        if (CSV_FILE_TEST == null)
            throw new RuntimeException("Test CSV file cannot be found");
        if (shape.volume != 784)
            throw new ShapeMismatchException("Invalid shape for MNIST data");

        System.out.println("Loading test data...");
        Tensor[][] testData = new Tensor[][]{new Tensor[10000], new Tensor[10000]};
        return loadCsvFile(new File(CSV_FILE_TEST.getFile()), testData, shape);
    }

    private Tensor[][] loadCsvFile(File file, Tensor[][] data, Shape shape){
        try {
            scanner = new Scanner(file);
        } catch (FileNotFoundException e) {
            System.out.println("CSV file cannot be found. Aborting.");
            System.exit(1);
        }
        scanner.nextLine(); // skip column names

        int id = 0;
        while (hasNext() && id < data[0].length) {
            data[0][id] = new Tensor(shape);
            data[1][id] = new Tensor(new Shape(10));
            List<String> line = getNext();
            for (int i = 1; i <= 28 * 28; i++){
                data[0][id].elements[i-1] = Float.parseFloat(line.get(i)) / 255;
            }

            data[1][id].elements[Integer.parseInt(line.get(0))] = 1;
            id++;
            if (id % 10000 == 0) {
                System.out.println(id + " records loaded total.");
            }
        }
        return data;
    }

    private boolean hasNext(){
        return scanner.hasNext();
    }

    private List<String> getNext(){
        if (hasNext()){
            return parseLine(scanner.nextLine());
        } else {
            scanner.close();
            return null;
        }
    }

    private static List<String> parseLine(String cvsLine) {
        List<String> result = new ArrayList<>();

        //if empty, return!
        if (cvsLine == null && cvsLine.isEmpty()) {
            return result;
        }

        StringBuffer curVal = new StringBuffer();
        boolean inQuotes = false;
        boolean startCollectChar = false;
        boolean doubleQuotesInColumn = false;

        char[] chars = cvsLine.toCharArray();

        for (char ch : chars) {

            if (inQuotes) {
                startCollectChar = true;
                if (ch == '"') {
                    inQuotes = false;
                    doubleQuotesInColumn = false;
                } else {

                    //Fixed : allow "" in custom quote enclosed
                    if (ch == '\"') {
                        if (!doubleQuotesInColumn) {
                            curVal.append(ch);
                            doubleQuotesInColumn = true;
                        }
                    } else {
                        curVal.append(ch);
                    }

                }
            } else {
                if (ch == '"') {

                    inQuotes = true;

                    //Fixed : allow "" in empty quote enclosed
                    if (chars[0] != '"') {
                        curVal.append('"');
                    }

                    //double quotes in column will hit this!
                    if (startCollectChar) {
                        curVal.append('"');
                    }

                } else if (ch == ',') {

                    result.add(curVal.toString());

                    curVal = new StringBuffer();
                    startCollectChar = false;

                } else if (ch == '\r') {
                    //ignore LF characters
                    continue;
                } else if (ch == '\n') {
                    //the end, break!
                    break;
                } else {
                    curVal.append(ch);
                }
            }

        }

        result.add(curVal.toString());

        return result;
    }

}
