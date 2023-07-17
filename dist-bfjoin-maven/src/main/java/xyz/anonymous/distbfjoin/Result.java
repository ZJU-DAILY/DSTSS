package xyz.anonymous.distbfjoin;

import java.io.Serializable;
import java.util.Objects;

public class Result implements Comparable<Result>, Serializable {

  public final Trajectory trajectory;
  public final double similarity;

  public Result(Trajectory trajectory, double similarity) {
    this.trajectory = trajectory;
    this.similarity = similarity;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    Result result = (Result) o;
    return Double.compare(result.similarity, similarity) == 0 && trajectory.equals(
        result.trajectory);
  }

  @Override
  public String toString() {
    return "Result{" +
        "trajectory=" + trajectory.getId() +
        ", similarity=" + similarity +
        '}';
  }

  @Override
  public int hashCode() {
    return Objects.hash(trajectory, similarity);
  }

  @Override
  public int compareTo(Result o) {
    return Double.compare(similarity, o.similarity);
  }
}
