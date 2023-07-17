package xyz.anonymous.distbfjoin;

import xyz.anonymous.distbfjoin.Trajectory.Cell;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.DoubleUnaryOperator;

public class UpperBoundTrajectorySimilarity implements TrajectorySimilarity {

  private final double alpha;
  /**
   * Normalize function for location distance
   */
  private final DoubleNormalizer nldf;
  /**
   * Normalize function for text distance
   */
  private final DoubleNormalizer ntdf;

  public UpperBoundTrajectorySimilarity(double alpha, DoubleNormalizer nldf,
      DoubleNormalizer ntdf) {
    this.alpha = alpha;
    this.nldf = nldf;
    this.ntdf = ntdf;
  }

  @Override
  public double similarity(Trajectory t1, Trajectory t2) {
    assert t1.getGridSize() == t2.getGridSize();
    double location = 2 * nldf.applyAsDouble(minDist(t1, t2) * t1.getGridSize());
    double text = ntdf.applyAsDouble(maxTSim(t1, t2));
    return alpha * location + (1 - alpha) * text;
  }

  private double minDist(Trajectory t1, Trajectory t2) {
    Set<Cell> aCells = t1.summarize().getCells();
    Set<Cell> bCells = t2.summarize().getCells();
    Set<Cell> union = new HashSet<>(aCells);
    union.addAll(bCells);
    List<Cell> Ax = new ArrayList<>(union);
    List<Cell> Ay = new ArrayList<>(union);
    Ax.sort(Comparator.comparingInt(Cell::getX));
    Ay.sort(Comparator.comparingInt(Cell::getY));
    return minDistUtil(aCells, bCells, Ax, Ay);
  }

  /**
   * @return minimal distance between c1 and c2, in terms of grid length
   */
  private double minDistUtil(Set<Cell> c1, Set<Cell> c2, List<Cell> Ax, List<Cell> Ay) {
    int m = Ax.size();
    if (m <= 3) {
      return minDistBruteForce(c1, c2, Ax);
    }
    int n = m / 2;
    List<Cell> Ax_l = Ax.subList(0, n);  // Cells to the left of Ax[n], sorted by x coordinate
    List<Cell> Ax_r = Ax.subList(n, m);  // Cells to the right of Ax[n] (inclusive), sorted by x
    List<Cell> Ay_l = new ArrayList<>();  // Cells to the left of Ax[n], sorted by y coordinate
    List<Cell> Ay_r = new ArrayList<>();  // Cells to the right of Ax[n] (inclusive), sorted by y
    for (Cell cell : Ay) {
      if (cell.getX() < Ax.get(n).getX()) {
        Ay_l.add(cell);
      } else {
        Ay_r.add(cell);
      }
    }
    double min_l = Double.MAX_VALUE;
    if (containsBoth(c1, c2, Ax_l)) {
      min_l = minDistUtil(c1, c2, Ax_l, Ay_l);
    }
    double min_r = Double.MAX_VALUE;
    if (containsBoth(c1, c2, Ax_r)) {
      min_r = minDistUtil(c1, c2, Ax_r, Ay_r);
    }
    double d_min = Math.min(min_l, min_r);
    List<Cell> Am = new ArrayList<>();  // cells "in the middle", sorted by y coordinate
    for (Cell cell : Ay) {
      if (cell.getX() - Ax.get(n).getX() < d_min) {
        Am.add(cell);
      }
    }
    // Compute the min distance between any two points from different trajectories in Am
    d_min = Math.min(d_min, minDistBruteForce(c1, c2, Am));
    return d_min;
  }

  /**
   * @return whether list `cells` contains cells from both c1 and c2
   */
  private boolean containsBoth(Set<Cell> c1, Set<Cell> c2, List<Cell> cells) {
    boolean has1 = false;
    boolean has2 = false;
    for (Cell cell : cells) {
      if (c1.contains(cell)) {
        has1 = true;
      }
      if (c2.contains(cell)) {
        has2 = true;
      }
      if (has1 && has2) {
        return true;
      }
    }
    return false;
  }

  /**
   * @return minimal distance between c1 and c2, in terms of grid length
   */
  private double minDistBruteForce(Set<Cell> c1, Set<Cell> c2, List<Cell> cells) {
    double d_min = Double.MAX_VALUE;
    for (int i = 0; i < cells.size(); i++) {
      for (int j = i + 1; j < cells.size(); j++) {
        if ((c1.contains(cells.get(i)) && c2.contains(cells.get(j)))
            || (c1.contains(cells.get(j)) && c2.contains(cells.get(i)))) {
          d_min = Math.min(d_min, cells.get(i).distance(cells.get(j)));
        }
      }
    }
    return d_min;
  }

  /**
   * @return text similarity upper bound
   */
  private double maxTSim(Trajectory t1, Trajectory t2) {
    double c1 = 0;
    for (Point p : t1) {
      if (containsOne(p.getTerms(), t2.summarize().getTerms())) {
        c1 += 1;
      }
    }
    double c2 = 0;
    for (Point p : t2) {
      if (containsOne(p.getTerms(), t1.summarize().getTerms())) {
        c2 += 1;
      }
    }
    return c1 / t1.length() + c2 / t2.length();
  }

  /**
   * @return whether any element in `list` is contained or `set`, or `list` is empty
   */
  private boolean containsOne(int[] list, Set<Integer> set) {
    if (list.length == 0) {
      return true;
    }
    for (int s : list) {
      if (set.contains(s)) {
        return true;
      }
    }
    return false;
  }
}
