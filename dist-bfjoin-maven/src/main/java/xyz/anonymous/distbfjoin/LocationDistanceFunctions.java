package xyz.anonymous.distbfjoin;

public class LocationDistanceFunctions {

  public static class EuclideanDistanceFunction implements DistanceFunction<double[]> {

    EuclideanDistanceFunction() {
    }

    @Override
    public double distance(double[] a, double[] b) {
      assert a.length == b.length;
      double sum = 0.0;
      for (int i = 0; i < a.length; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
      }
      return Math.sqrt(sum);
    }
  }

  public static EuclideanDistanceFunction EUCLIDEAN_DISTANCE_FUNCTION = new EuclideanDistanceFunction();
}
