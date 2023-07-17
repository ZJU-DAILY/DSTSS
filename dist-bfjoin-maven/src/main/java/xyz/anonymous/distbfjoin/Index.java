package xyz.anonymous.distbfjoin;

import java.util.Collection;
import java.util.List;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * An index that supports trajectory similarity search.
 */
public interface Index {

  /**
   * Adds a trajectory to the index.
   * @param trajectory the trajectory
   */
  void add(Trajectory trajectory);

  default void addAll(Collection<Trajectory> trajectories) {
    Logger logger = LogManager.getLogger(Index.class);
    logger.info("Adding trajectories to the index");
    int count = 0;
    for (Trajectory t : trajectories) {
      count++;
      if (count % 10000 == 0) {
        logger.debug("{} trajectories added", count);
      }
      add(t);
    }
  }

  void setTheta(double theta);

  /**
   * Return k trajectories in the index that are most similar to the query.
   * @param query the query trajectory
   * @param k the number of trajectories to return
   * @return a list of k most similar trajectories with their similarity scores
   */
  List<Result> search(Trajectory query, int k);
}
