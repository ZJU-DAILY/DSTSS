package xyz.anonymous.distbfjoin;

import static xyz.anonymous.distbfjoin.LocationDistanceFunctions.EUCLIDEAN_DISTANCE_FUNCTION;
import static java.util.Comparator.naturalOrder;
import static java.util.stream.Collectors.toList;

import xyz.anonymous.distbfjoin.bfjoin.MBR;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;

import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;
import java.io.Serializable;
import java.util.*;
import java.util.stream.IntStream;

public class Trajectory implements Iterable<Point>, Serializable {

  private int id;
  private final List<Point> points = new ArrayList<>();
  transient private TrajectorySummary summary = null;
  private final double gridSize;

  public Trajectory() {
    this(-1, 0.01);
  }

  public Trajectory(double gridSize) {
    this(-1, gridSize);
  }

  public Trajectory(int id, double gridSize) {
    this.id = id;
    this.gridSize = gridSize;
  }

  public Trajectory(Collection<Point> points, int id, double gridSize) {
    this.points.addAll(points);
    this.id = id;
    this.gridSize = gridSize;
  }

  public Trajectory(Collection<Point> points, int id) {
    this(points, id, 0.01);
  }

  public Trajectory(Collection<Point> points) {
    this(points, -1, 0.01);
  }

  public int getId() {
    return id;
  }

  public Trajectory setId(int id) {
    this.id = id;
    return this;
  }

  public List<double[]> getLocations() {
    return points.stream().map(Point::getLocation).collect(toList());
  }

  /**
   * Adds a point.
   * @param p the new point
   * @return this trajectory
   */
  public Trajectory addPoint(Point p) {
    this.points.add(p);
    return this;
  }

  public Trajectory addPoints(Collection<Point> points) {
    this.points.addAll(points);
    return this;
  }

  public List<Point> getPoints() {
    return points;
  }

  /**
   * Sorts the points.
   * @return this trajectory
   */
  public Trajectory sort() {
    this.points.sort(naturalOrder());
    return this;
  }

  public double getGridSize() {
    return gridSize;
  }

  public int length() {
    return points.size();
  }

  /**
   * Get the description terms (with duplicates).
   *
   * @return the terms
   */
  public Iterable<Integer> getTermIterator() {
    return () -> new Iterator<Integer>() {
      private final Iterator<Point> pit = points.iterator();
      private Iterator<Integer> sit = Arrays.stream(pit.next().getTerms()).iterator();

      @Override
      public boolean hasNext() {
        while (!sit.hasNext()) {
          if (!pit.hasNext()) {
            return false;
          }
          sit = Arrays.stream(pit.next().getTerms()).iterator();
        }
        return true;
      }

      @Override
      public Integer next() {
        while (!sit.hasNext()) {
          sit = Arrays.stream(pit.next().getTerms()).iterator();
        }
        return sit.next();
      }
    };
  }

  @Override
  public Iterator<Point> iterator() {
    return points.iterator();
  }

  public Trajectory subTrajectory(int sample) {
    if (sample <= this.points.size()) {
      return this;
    }
    List<Integer> indexes = IntStream.range(0, this.points.size()).boxed()
        .collect(toList());
    Collections.shuffle(indexes);
    indexes = indexes.subList(0, sample);
    indexes.sort(naturalOrder());

    List<Point> newPoints = indexes.stream().map(this.points::get).collect(toList());
    newPoints = newPoints.subList(0, sample);
    return new Trajectory(newPoints, id);
  }

  public TrajectorySummary summarize() {
    if (summary == null) {
      summary = new TrajectorySummary();
    }
    return summary;
  }

  @Override
  public String toString() {
    StringBuilder str = new StringBuilder("Trajectory{id=").append(id).append(", points=[");
    Iterator<Point> it = points.iterator();
    while (it.hasNext()) {
      str.append("\n  ").append(it.next());
      if (it.hasNext()) {
        str.append(",");
      }
    }
    str.append("\n]}");
    return str.toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    Trajectory points = (Trajectory) o;
    return id == points.id;
  }

  @Override
  public int hashCode() {
    return Objects.hash(id);
  }

  public class TrajectorySummary implements Serializable {

    private final MBR mbr;
    private final Set<Cell> cells = new ObjectOpenHashSet<>();
    private final Set<Integer> termSummary;

    private TrajectorySummary() {
      this.mbr = MBR.of(points.stream().map(Point::getLocation).collect(toList()));
      for (Point point : points) {
        cells.add(new Cell(point.getLocation(), gridSize));
      }
      termSummary = new IntOpenHashSet();
      for (int term : getTermIterator()) {
        termSummary.add(term);
      }
    }

    public MBR getMbr() {
      return mbr;
    }

    public Set<Cell> getCells() {
      return cells;
    }

    public Set<Integer> getTerms() {
      return termSummary;
    }
  }

  public static class Cell implements Serializable {

    private final static DistanceFunction<double[]> df = EUCLIDEAN_DISTANCE_FUNCTION;

    private final int x;
    private final int y;

    public Cell(double[] location, double gridSize) {
      assert location.length == 2;
      this.x = (int) Math.floor(location[0] / gridSize);
      this.y = (int) Math.floor(location[1] / gridSize);
    }

    public int getX() {
      return x;
    }

    public int getY() {
      return y;
    }

    /**
     * Calculates the distance to the other cell in terms of grid length.
     *
     * @param that the other cell
     * @return the distance, in terms of grid length
     */
    public double distance(Cell that) {
      // Where is `that` w.r.t. `this`?
      boolean left = that.x < this.x - 1;
      boolean right = that.x > this.x + 1;
      boolean top = that.y > this.y + 1;
      boolean bottom = that.y < this.y - 1;

      if (left && bottom) {
        return df.distance(new double[]{this.x, this.y}, new double[]{that.x + 1, that.y + 1});
      }
      if (left && top) {
        return df.distance(new double[]{this.x, this.y + 1}, new double[]{that.x + 1, that.y});
      }
      if (right && bottom) {
        return df.distance(new double[]{this.x + 1, this.y}, new double[]{that.x, that.y + 1});
      }
      if (right && top) {
        return df.distance(new double[]{this.x + 1, this.y + 1}, new double[]{that.x, that.y});
      }
      if (left) {
        return this.x - that.x - 1;
      }
      if (right) {
        return that.x - this.x - 1;
      }
      if (top) {
        return that.y - this.y - 1;
      }
      if (bottom) {
        return this.y - that.y - 1;
      }
      return 0;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      Cell cell = (Cell) o;
      return x == cell.x && y == cell.y;
    }

    @Override
    public int hashCode() {
      return Objects.hash(x, y);
    }
  }
}
