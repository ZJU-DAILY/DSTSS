package xyz.anonymous.distbfjoin;

import java.io.Serializable;
import java.util.List;
import java.util.function.DoubleUnaryOperator;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class ExactTrajectorySimilarity implements TrajectorySimilarity, Serializable {

  private static final Logger logger = LogManager.getLogger(ExactTrajectorySimilarity.class);

  private final double alpha;
  /**
   * Location distance function
   */
  private final DistanceFunction<double[]> ldf;
  /**
   * Text distance function
   */
  private final DistanceFunction<int[]> tdf;
  /**
   * Normalize function for location distance
   */
  private final DoubleNormalizer nldf;
  /**
   * Normalize function for text distance
   */
  private final DoubleNormalizer ntdf;

  /**
   * Creates an exact similarity calculator.
   *
   * @param alpha the balance between location similarity and text similarity
   * @param ldf   location distance function
   * @param tdf   text similarity function
   * @param nldf  normalize function for location distance
   * @param ntdf  normalize function for text distance
   */
  public ExactTrajectorySimilarity(double alpha, DistanceFunction<double[]> ldf,
      DistanceFunction<int[]> tdf, DoubleNormalizer nldf,
      DoubleNormalizer ntdf) {
    this.alpha = alpha;
    this.ldf = ldf;
    this.tdf = tdf;
    this.nldf = nldf;
    this.ntdf = ntdf;
  }

  /**
   * @param a trajectory a
   * @param b trajectory b
   * @return similarity
   */
  @Override
  public double similarity(Trajectory t1, Trajectory t2) {
    double s1 = 0;
    for (Point p : t1) {
      s1 += relevance(p, t2);
    }
    s1 /= t1.length();
    double s2 = 0;
    for (Point p : t2) {
      s2 += relevance(p, t1);
    }
    s2 /= t2.length();
    return s1 + s2;
  }

  private double relevance(Point o, Trajectory t) {
    double result = 0;
    for (Point p : t) {
      result = Math.max(result, similarity(o, p));
    }
    return result;
  }

  private double similarity(Point x, Point y) {
    double ld = ldf.distance(x.getLocation(), y.getLocation());
    double td = tdf.distance(x.getTerms(), y.getTerms());
    double nld = alpha * nldf.applyAsDouble(ld);
    double ntd = (1 - alpha) * ntdf.applyAsDouble(td);
    logger.trace("point similarity: {} + {}", nld, ntd);
    return nld + ntd;
  }
}
