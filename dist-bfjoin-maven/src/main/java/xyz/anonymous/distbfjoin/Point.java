package xyz.anonymous.distbfjoin;

import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

import xyz.anonymous.distbfjoin.utils.StringRecord;
import org.jetbrains.annotations.NotNull;

/**
 * An "object" in the trajectory.
 */
public class Point implements Comparable<Point>, Serializable {

  private static StringRecord record = new StringRecord();
  private final long pid;  // smaller pid appears first in a trajectory
  private final double[] location;
  private final int[] terms;

  static {
    record.preread("/home/anonymous/trajectories/semantics/keywords.txt");
    record.preread("/home/anonymous/trajectories/semantics/amap-labels.txt");
  }

  public Point(long pid, double x, double y, String[] terms) {
    this(pid, new double[]{x, y}, terms != null ? Arrays.asList(terms) : null);
  }

  public Point(long pid, double[] location, List<String> terms) {
    if (record == null) {
      throw new IllegalStateException("StringRecord is not set.");
    }
    this.pid = pid;
    this.location = location;
    if (terms == null) {
      this.terms = new int[0];
    } else {
      this.terms = terms.stream().mapToInt(record::get).toArray();
    }
  }

  public double[] getLocation() {
    return location;
  }

  /**
   * Gets the description, which is a list of terms.
   * @return the description
   */
  public int[] getTerms() {
    return terms;
  }

  @Override
  public int compareTo(@NotNull Point o) {
    return Long.compare(this.pid, o.pid);
  }

  @Override
  public String toString() {
    return "Point{" +
        "pid=" + pid +
        ", location=" + Arrays.toString(location) +
        ", terms=" + Arrays.toString(terms) +
        '}';
  }
}
