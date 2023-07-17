package anonymous.trajectory;


import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.CoreMatchers.is;

import com.github.jelmerk.knn.DistanceFunction;
import com.github.jelmerk.knn.DistanceFunctions;
import com.github.jelmerk.knn.Item;
import com.github.jelmerk.knn.SearchResult;
import com.github.jelmerk.knn.hnsw.HnswIndex;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BinaryOperator;
import java.util.stream.Collectors;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class HnswIndexSpatialTest {

  static class DTItem implements Item<String, double[]> {

    private static final long serialVersionUID = 1L;

    private final String id;
    private final double[] vector;
    private final long version;

    public DTItem(String id, double[] vector) {
      this(id, vector, 0);
    }

    public DTItem(String id, double[] vector, long version) {
      this.id = id;
      this.vector = vector;
      this.version = version;
    }

    @Override
    public String id() {
      return id;
    }

    @Override
    public double[] vector() {
      return vector;
    }

    @Override
    public long version() {
      return version;
    }

    @Override
    public int dimensions() {
      return vector.length;
    }

    @Override
    public String toString() {
      return "TestItem{" + "id='" + id + '\'' + ", vector=" + Arrays.toString(vector) + ", version=" + version + '}';
    }
  }

  static class DoubleDistanceFunctionWithCount implements DistanceFunction<double[], Double> {

    private final DistanceFunction<double[], Double> distanceFunction;
    private final AtomicInteger count = new AtomicInteger();

    public DoubleDistanceFunctionWithCount(DistanceFunction<double[], Double> distanceFunction) {
      this.distanceFunction = distanceFunction;
    }

    @Override
    public Double distance(double[] u, double[] v) {
      count.getAndIncrement();
      return this.distanceFunction.distance(u, v);
    }

    public int getCount() {
      return count.getAndSet(0);
    }
  }

  static private final int maxItemCount = 30000;
  private final int m = 10;
  private final int efConstruction = 250;
  private final int ef = 20;
  static private final int dimensions = 300;
  private final int maxLevel = 3;
  private final DoubleDistanceFunctionWithCount distanceFunction = new DoubleDistanceFunctionWithCount(DistanceFunctions.DOUBLE_EUCLIDEAN_DISTANCE);
  private final DoubleDistanceFunctionWithCount constrDistanceFunction = new DoubleDistanceFunctionWithCount(DistanceFunctions.DOUBLE_EUCLIDEAN_DISTANCE);
  private final BinaryOperator<Double> differenceCalculator = DifferenceCalculators.DOUBLE_DIFFERENCE_CALCULATOR;
  private HnswIndexSpatial<String, double[], DTItem, Double> index;
  private HnswIndexSpatial2<String, double[], DTItem, Double> index2;
  private HnswIndex<String, double[], DTItem, Double> hnswIndex;
  static private List<DTItem> items;
  static private final int queryCount = 30000;
  static private List<double[]> queries;

  @BeforeAll
  static void read() throws IOException {
    assert dimensions == 300;

    DistanceFunction<double[], Double> df = DistanceFunctions.DOUBLE_EUCLIDEAN_DISTANCE;
    DataSampler sampler1 = new DataSampler("");
    AtomicInteger i = new AtomicInteger();
    items = sampler1.sample(maxItemCount).stream().map(array -> new DTItem(Integer.toString(i.getAndIncrement()), array)).collect(
        Collectors.toList());

    DataSampler sampler2 = new DataSampler("");
    queries = sampler2.sample(queryCount);
  }

  void setUp() throws IOException, InterruptedException {
    index = HnswIndexSpatial.newBuilder(dimensions, distanceFunction, constrDistanceFunction, differenceCalculator,
                                         maxItemCount).withM(m).withEf(ef).withEfConstruction(efConstruction)
        .withMaxLevel(maxLevel).<String, DTItem>refine().withEntryPointPicker((item) -> item.id().equals("3"))
        .withEntryPointGetters((vector) -> items.get(3), (vector) -> items.get(3)).build();
    Instant start = Instant.now();
    index.addAll(items);
    Instant finish = Instant.now();
    long timeElapsed = Duration.between(start, finish).toMillis();
    System.out.printf("index construction time: %d ms\n", timeElapsed);
    countDist();
  }

  void setUp2() throws IOException, InterruptedException {

    index2 = HnswIndexSpatial2.newBuilder(dimensions, distanceFunction, constrDistanceFunction, differenceCalculator,
                                          maxItemCount).withM(m).withEf(ef).withEfConstruction(efConstruction)
        .withMaxLevel(maxLevel).<String, DTItem>refine().withEntryPointPicker((item) -> item.id().equals("3"))
        .withEntryPointGetters((vector) -> items.get(3), (vector) -> items.get(3)).build();
    Instant start = Instant.now();
    index2.addAll(items);
    Instant finish = Instant.now();
    long timeElapsed = Duration.between(start, finish).toMillis();
    System.out.printf("index construction time: %d ms\n", timeElapsed);
    countDist2();
  }

  void setUpHnsw() throws IOException, InterruptedException {
    hnswIndex = HnswIndex.newBuilder(dimensions, distanceFunction, maxItemCount).withM(m).withEf(ef)
        .withEfConstruction(efConstruction).build();
    Instant start = Instant.now();
    hnswIndex.addAll(items);
    Instant finish = Instant.now();
    long timeElapsed = Duration.between(start, finish).toMillis();
    System.out.printf("hnswIndex construction time: %d ms\n", timeElapsed);
  }

  void countDist() throws IOException {
//    Files.write(Paths.get("src/test/java/cn/edu/zju/daily/vector/index.txt"), index.toString().getBytes());
    System.out.printf("distance count: %d\n", distanceFunction.getCount());
    System.out.printf("constr distance count: %d\n", constrDistanceFunction.getCount());
  }

  void countDist2() throws IOException {
//    Files.write(Paths.get("src/test/java/cn/edu/zju/daily/vector/index.txt"), index2.toString().getBytes());
    System.out.printf("distance count: %d\n", distanceFunction.getCount());
    System.out.printf("constr distance count: %d\n", constrDistanceFunction.getCount());
  }

  void tearDownHnsw() throws IOException {
//    Files.write(Paths.get("src/test/java/cn/edu/zju/daily/vector/indexHnsw.txt"), hnswIndex.toString().getBytes());
    System.out.printf("distance count: %d\n", distanceFunction.getCount());
    System.out.printf("constr distance count: %d\n", constrDistanceFunction.getCount());
  }

  void testHnsw() throws IOException, InterruptedException {
    setUpHnsw();
    Instant start = Instant.now();
    for (int i = 0; i < queryCount; i++) {
      List<SearchResult<DTItem, Double>> results = hnswIndex.findNearest(queries.get(i), 11);
    }
    Instant finish = Instant.now();
    long timeElapsed = Duration.between(start, finish).toMillis();
    System.out.printf("hnsw search time: %d ms\n", timeElapsed);
    tearDownHnsw();
  }

  @Test
  void test() throws IOException, InterruptedException {
    System.out.println("------- HnswIndexSpatial -------");
    setUp();
    Instant start = Instant.now();
    for (int i = 0; i < queryCount; i++) {
      List<SearchResult<DTItem, Double>> results = index.findNearest(queries.get(i), 11);
    }
    Instant finish = Instant.now();
    long timeElapsed = Duration.between(start, finish).toMillis();
    System.out.printf("hnsw search time: %d ms\n", timeElapsed);
    countDist();
  }
  void testQuerySelf() throws IOException, InterruptedException {
    System.out.println("------- HnswIndexSpatial (query self nodes) -------");
    setUp();
    Instant start = Instant.now();
    for (int i = 0; i < maxItemCount; i++) {
      List<SearchResult<DTItem, Double>> results = index.findNearest(items.get(i).vector(), 11);
    }
    Instant finish = Instant.now();
    long timeElapsed = Duration.between(start, finish).toMillis();
    System.out.printf("hnsw search time: %d ms\n", timeElapsed);
    countDist();
  }


  @Test
  void test2() throws IOException, InterruptedException {
    System.out.println("------- HnswIndexSpatial2 (using d2) -------");
    setUp2();
    Instant start = Instant.now();
    for (int i = 0; i < queryCount; i++) {
      List<SearchResult<DTItem, Double>> results = index2.findNearest(queries.get(i), 11);
    }
    Instant finish = Instant.now();
    long timeElapsed = Duration.between(start, finish).toMillis();
    System.out.printf("hnsw search time: %d ms\n", timeElapsed);
    countDist2();
  }


  void testQuerySelf2() throws IOException, InterruptedException {
    System.out.println("------- HnswIndexSpatial2 (optimization disabled, query self nodes) -------");
    setUp2();
    Instant start = Instant.now();
    for (int i = 0; i < maxItemCount; i++) {
      List<SearchResult<DTItem, Double>> results = index2.findNearest(items.get(i).vector(), 11);
    }
    Instant finish = Instant.now();
    long timeElapsed = Duration.between(start, finish).toMillis();
    System.out.printf("hnsw search time: %d ms\n", timeElapsed);
    countDist2();
  }

//  void testBoth() throws IOException, InterruptedException {
//    setUp();
//    setUpHnsw();
//    double[] arr = new double[dimensions];
//    for (int i = 0; i < queryCount; i++) {
//      for (int d = 0; d < dimensions; d++) {
//        arr[d] = ThreadLocalRandom.current().nextDouble(-0.2, 0.2);
//      }
//      List<SearchResult<DTItem, Double>> results = index.findNearest(arr, 10);
//      List<SearchResult<DTItem, Double>> hnswResults = hnswIndex.findNearest(arr, 10);
//      assertThat(results, is(hnswResults));
//    }
  // tearDown();
  // tearDownHnsw();
//  }
}
