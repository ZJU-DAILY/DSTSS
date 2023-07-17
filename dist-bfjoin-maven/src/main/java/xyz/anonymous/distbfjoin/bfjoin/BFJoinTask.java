package xyz.anonymous.distbfjoin.bfjoin;

import xyz.anonymous.distbfjoin.DoubleNormalizer;
import xyz.anonymous.distbfjoin.Point;
import xyz.anonymous.distbfjoin.Result;
import xyz.anonymous.distbfjoin.Trajectory;
import xyz.anonymous.distbfjoin.TrajectorySimilarity;
import xyz.anonymous.distbfjoin.utils.Batch;
import xyz.anonymous.distbfjoin.utils.Timer;

import java.io.Serializable;
import java.util.List;
import java.util.PriorityQueue;
import java.util.function.DoubleUnaryOperator;
import scala.collection.mutable.ArrayBuffer;
import xyz.anonymous.distbfjoin.*;
import xyz.anonymous.distbfjoin.utils.Batch;
import xyz.anonymous.distbfjoin.utils.Timer;


public class BFJoinTask implements Serializable {

  private final double alpha;
  private final TrajectorySimilarity uts;
  private final TrajectorySimilarity ts;
  private final DoubleNormalizer normalizer;

  /**
   * Creates a BFJoin task.
   *
   * @param alpha      the balance between location similarity and text similarity
   * @param uts        the upper bound trajectory similarity function
   * @param ts         the exact trajectory similarity function
   * @param normalizer the distance normalizer (to [0, 1])
   */
  public BFJoinTask(double alpha, TrajectorySimilarity uts, TrajectorySimilarity ts,
      DoubleNormalizer normalizer) {

    this.alpha = alpha;
    this.uts = uts;
    this.ts = ts;
    this.normalizer = normalizer;
  }

  /**
   * @param query   the query trajectory
   * @param batches the list of batches
   * @return a list of results
   */
  public TaskResult call(Trajectory query, List<Batch> batches, int k) {
    Timer timer = new Timer();

    PriorityQueue<Result> results = new PriorityQueue<>();
    double theta = 0;
    for (Batch batch : batches) {
      // query-batch filtering (stage 1) (intra-list batch filtering phase)
      double coarseUpperBound = getCoarseUpperBound(query, batch);
      if (coarseUpperBound < theta) {
        continue;
      }

      // query-batch filtering (stage 2) (trajectory-batch filtering phase)
      double fineUpperBound = getFineUpperBound(query, batch, coarseUpperBound);
      assert fineUpperBound < coarseUpperBound;
      if (fineUpperBound < theta) {
        continue;
      }

      for (Trajectory t : batch) {
        double upperBound = uts.similarity(query, t);
        if (upperBound < theta) {
          continue;
        }
        double similarity = ts.similarity(query, t);
        if (similarity >= theta) {
          results.add(new Result(t, similarity));
          if (results.size() > k) {
            results.poll();
            Result peek = results.peek();
            assert peek != null;
            theta = peek.similarity;
          }
        }
      }
    }

    return new TaskResult(results, timer.duration(), 1);
  }

  private double getCoarseUpperBound(Trajectory t, Batch b) {
    double minDist = b.getMbr().distance(t.summarize().getMbr());
    double normalizedDist = normalizer.applyAsDouble(minDist);
    return 2.0 * (alpha * normalizedDist + (1 - alpha));
  }

  private double getFineUpperBound(Trajectory t, Batch b, double coarseUpperBound) {
    double sum = 0;
    for (Point p : t) {
      sum += getUpperBoundRelevance(p, b);
    }
    sum /= t.length();
    return sum + coarseUpperBound / 2;
  }

  private double getUpperBoundRelevance(Point p, Batch b) {
    double minDist = b.getMbr().distance(p.getLocation());
    int[] terms = p.getTerms();
    int all = terms.length;
    if (all == 0) {
      return alpha * normalizer.applyAsDouble(minDist) + (1 - alpha);
    }
    int remaining = 0;
    for (int term : terms) {
      if (b.hasTerm(term)) {
        remaining++;
      }
    }
    return alpha * normalizer.applyAsDouble(minDist) + (1 - alpha) * ((double) remaining / all);
  }
}
