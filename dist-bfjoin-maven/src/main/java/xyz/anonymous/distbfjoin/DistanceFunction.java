package xyz.anonymous.distbfjoin;

import java.io.Serializable;

public interface DistanceFunction<T> extends Serializable {

  double distance(T a, T b);
}
