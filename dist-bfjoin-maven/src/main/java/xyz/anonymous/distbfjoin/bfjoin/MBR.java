package xyz.anonymous.distbfjoin.bfjoin;

import static xyz.anonymous.distbfjoin.LocationDistanceFunctions.EUCLIDEAN_DISTANCE_FUNCTION;

import xyz.anonymous.distbfjoin.DistanceFunction;
import xyz.anonymous.distbfjoin.DistanceFunction;
import xyz.anonymous.distbfjoin.LocationDistanceFunctions;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * This class represents a 2D MBR. Euclidean distance is used to calculate the distance between two
 * MBRs and between an MBR and a point.
 */
public final class MBR implements Serializable {

  private final static double[] VOID = new double[]{Double.NaN, Double.NaN};
  private final static DistanceFunction<double[]> df = LocationDistanceFunctions.EUCLIDEAN_DISTANCE_FUNCTION;

  private final double[] min;
  private final double[] max;

  /**
   * An MBR that contains nothing.
   */
  public MBR() {
    this(VOID, VOID);
  }

  /**
   * Copy initializer.
   *
   * @param other the other MBR
   */
  public MBR(MBR other) {
    this(other.min, other.max);
  }

  public MBR(double[] min, double[] max) {
    this.min = Arrays.copyOf(min, min.length);
    this.max = Arrays.copyOf(max, max.length);
  }

  public boolean isInfinitesimal() {
    return Arrays.equals(VOID, min) /* && Arrays.equals(VOID, max) */;
  }

  /**
   * Get diagonal distance.
   *
   * @return the diagonal distance
   */
  public double getDiag() {
    if (isInfinitesimal()) {
      return 0;
    }
    return df.distance(min, max);
  }

  /**
   * Get the area.
   *
   * @return the area
   */
  public double getArea() {
    if (isInfinitesimal()) {
      return 0;
    }
    double result = 1;
    for (int i = 0; i < 2; i++) {
      result *= max[i] - min[i];
    }
    return result;
  }


  /**
   * Returns a new MBR that is the union of the original MBR and the point.
   *
   * @param point the point
   * @return the new MBR
   */
  public MBR include(double[] point) {
    assert point.length == 2;

    if (isInfinitesimal()) {
      return new MBR(point, point);
    } else {
      MBR result = new MBR(this);
      for (int i = 0; i < 2; i++) {
        result.min[i] = Math.min(result.min[i], point[i]);
        result.max[i] = Math.max(result.max[i], point[i]);
      }
      return result;
    }
  }

  /**
   * Returns a new MBR that is the union of this MBR and the other MBR.
   *
   * @param other the other MBR
   * @return the new MBR
   */
  public MBR include(MBR other) {
    if (this.isInfinitesimal()) {
      return other;
    }
    if (other.isInfinitesimal()) {
      return this;
    }
    MBR result = new MBR(this);
    for (int i = 0; i < 2; i++) {
      result.min[i] = Math.min(result.min[i], other.min[i]);
      result.max[i] = Math.max(result.max[i], other.max[i]);
    }
    return result;
  }

  /**
   * Adds the new point to this MBR. This changes the current MBR and is only used internally.
   *
   * @param point the point
   */
  private void add(double[] point) {
    assert point.length == 2;

    if (isInfinitesimal()) {
      for (int i = 0; i < 2; i++) {
        min[i] = point[i];
        max[i] = point[i];
      }
    } else {
      for (int i = 0; i < 2; i++) {
        min[i] = Math.min(min[i], point[i]);
        max[i] = Math.max(max[i], point[i]);
      }
    }
  }

  /**
   * Get the minimal Euclidean distance between two MBRs.
   *
   * @param that the other MBR
   * @return the minimal distance
   */
  public double distance(MBR that) {
    double distance = _distance(that);
    assert distance >= 0;
    return distance;
  }

  private double _distance(MBR that) {
    /* Where is `that` w.r.t. `this`? */
    boolean left = that.max[0] <= this.min[0];
    boolean right = this.max[0] <= that.min[0];
    boolean bottom = that.max[1] <= this.min[1];
    boolean top = this.max[1] <= that.min[1];

    if (left && bottom) {
      return df.distance(that.max, this.min);
    }
    if (right && top) {
      return df.distance(this.max, that.min);
    }
    if (left && top) {
      return df.distance(new double[]{that.max[0], that.min[1]},
          new double[]{this.min[0], this.max[1]});
    }
    if (right && bottom) {
      return df.distance(new double[]{this.max[0], this.min[1]},
          new double[]{that.min[0], that.max[1]});
    }
    if (left) {
      return this.min[0] - that.max[0];
    }
    if (right) {
      return that.min[0] - this.max[0];
    }
    if (top) {
      return that.min[1] - this.max[1];
    }
    if (bottom) {
      return this.min[1] - that.max[1];
    }
    // Overlap
    return 0;
  }

  /**
   * Get the minimal Euclidean distance to a point.
   *
   * @param point the point
   * @return the minimal distance
   */
  public double distance(double[] point) {
    return distance(new MBR(point, point));
  }

  public static MBR of(List<double[]> points) {
    MBR mbr = new MBR();
    for (double[] point : points) {
      mbr.add(point);
    }
    return mbr;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    MBR mbr = (MBR) o;
    return Arrays.equals(min, mbr.min) && Arrays.equals(max, mbr.max);
  }

  @Override
  public int hashCode() {
    int result = Arrays.hashCode(min);
    result = 31 * result + Arrays.hashCode(max);
    return result;
  }
}
