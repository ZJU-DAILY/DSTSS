package xyz.anonymous.distbfjoin;

import java.io.Serializable;

/**
 * A function to calculate the similarity of two trajectories.
 */
public interface TrajectorySimilarity extends Serializable {

  /**
   * Calculates the similarity between trajectories.
   * @param a trajectory a
   * @param b trajectory b
   * @return the similarity between trajectories
   */
  double similarity(Trajectory a, Trajectory b);
}
