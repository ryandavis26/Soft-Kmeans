import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
public class KMeans {


    public static void main(String[] args) throws IOException {
        /*
        Runs the K Means algorithm using the Rosalind formatted text
         */
        FileReader fr = new FileReader("SofftKMeans_Rosalind_Example.txt");
        BufferedReader br = new BufferedReader(fr);
        String km = br.readLine();
        int k =  Integer.parseInt(km.substring(0,1));
        int m =  Integer.parseInt(km.substring(2));
        String betaString = br.readLine();
        double beta = Double.parseDouble(betaString);
        ArrayList<ArrayList<Double>> data = new ArrayList<>();
        while (br.ready()){
            String curLn = br.readLine();
            String[] splitCurLn = curLn.trim().split("\\s+");
            ArrayList<Double> curPoint = new ArrayList<Double>();
            for (int i = 0; i < m; i++) {
                double point = Double.parseDouble(splitCurLn[i]);
                curPoint.add(i,point);
            }
            data.add(curPoint);
        }
        System.out.println();
        ArrayList<ArrayList<Double>> centers = kMeans(data, k, beta, 100);
        for (ArrayList<Double> center:centers) {
            for (double value:center) {
                System.out.printf("%.3f" + " ", value);
            }
            System.out.println();
        }
    }

    /*
        Takes a 2D list of on arbitrarily large set of points, and cluster them into k number of clusters using stiffness parameters
        Beta, and the number of iterations as chosen by the user
        Returns a List of Centers, where a center is m-dimensional number of doubles in a row
     */
    public static ArrayList<ArrayList<Double>> kMeans(ArrayList<ArrayList<Double>> points, int k_cluster, double beta, int numberOfIteration){
        ArrayList<ArrayList<Double>> centers = new ArrayList<>();
        for (int i = 0; i < k_cluster; i++) {
            centers.add(points.get(i));
        }
        for (int j = 0; j < numberOfIteration; j++) {
            double[][] hidMatrix = hiddenMatrix(points, centers, beta);

            centers = updateCenters(hidMatrix, points);
        }
        return centers;
    }

    /*
    Calculate the Euclidean Distance Between 2 Lists of Points
    This is a generalized version to handle 2 or more points in as a vector list
     */
    public static double euclidean_distance(ArrayList<Double> firstPoint, ArrayList<Double> secondPoint){
        double distance = 0.0;
        for (int i = 0; i < firstPoint.size(); i++) {
            distance = distance + Math.pow(firstPoint.get(i) - secondPoint.get(i), 2);
        }
        distance = Math.sqrt(distance);
        return distance;
    }

    /*
    Calculates the Hidden Matrix for the Soft K-means Center Distance Update
    Still not all too sure, this hidden matrix seems like some straight up magic, but the equations are
    fairly simple to implement
    Creates a Matrix of size k x n, watch the Youtube videos here to learn/remind yourself
    https://www.youtube.com/watch?v=xtDMHPVDDKk&list=PLQ-85lQlPqFMcC2d2CkvmdcJt2v-Np7Cz&index=
    https://www.youtube.com/watch?v=fpM0iZTjLhM&list=PLQ-85lQlPqFMcC2d2CkvmdcJt2v-Np7Cz&index=8
    This Matrix assigns a responsibility between data points and all centers, the more liklely a point belongs
    to a center the value's corresponding index will be
    My (Ryan Davis) implementation took some inspiration from a python implementation of soft k-means Hidden Matrix
     */
    public static double[][] hiddenMatrix(ArrayList<ArrayList<Double>> pointsList, ArrayList<ArrayList<Double>> centers, double beta){
        double[][] hidMatrix  = new double[centers.size()][pointsList.size()];
        for (int j = 0; j < pointsList.size(); j++) {
            double sumAllCenters = 0;
            for (ArrayList<Double> center : centers) {
                sumAllCenters = sumAllCenters + Math.exp(-beta * euclidean_distance(center, pointsList.get(j)));
            }
            for (int i = 0; i < centers.size(); i++) {
                hidMatrix[i][j] = Math.exp(-beta * euclidean_distance(centers.get(i), pointsList.get(j))) / sumAllCenters;
            }
        }
        return hidMatrix;
    }

    /*
    Returns the summation over all values from an array, a double in this case (Although this should be made generic
    using Java Generics)
     */
    public static double sumOverArray(double[] oneDMatrix){
        double sum = 0.0;
        for (double value: oneDMatrix) {
            sum += value;
        }
        return sum;
    }

    /*
    Creates a List of List of m sized points all initialized with all double values initialized to zero
     */
    public static ArrayList<ArrayList<Double>> initialize2DArrayList(int sizeOuter, int sizeInner){
        ArrayList<ArrayList<Double>> input = new ArrayList<ArrayList<Double>>();
        for (int i = 0; i < sizeOuter; i++) {
            ArrayList<Double> innerList = new ArrayList<Double>();
            for (int j = 0; j < sizeInner; j++) {
                innerList.add(0.0);
            }
            input.add(i, innerList);
        }
        return input;
    }
    /*
    Updates the list of centers by looking at the centers distance from the points
    Calculates the probability that a center belongs to its nearby points using the hidden matrix value
     */
    public static ArrayList<ArrayList<Double>> updateCenters(double[][] hidMatrix, ArrayList<ArrayList<Double>> points){
        int k = hidMatrix.length;int n = points.size();int m = points.get(0).size();

        ArrayList<ArrayList<Double>> new_centers = initialize2DArrayList(k,m);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < m; j++) {
                double product = 0;
                for (int l = 0; l < n; l++) {
                    product = product + points.get(l).get(j) * hidMatrix[i][l];
                }
                new_centers.get(i).set(j,product / sumOverArray(hidMatrix[i]));
            }
        }
        return new_centers;
    }
}



