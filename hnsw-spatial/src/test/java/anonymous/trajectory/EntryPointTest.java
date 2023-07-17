package anonymous.trajectory;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.CoreMatchers.is;

import com.github.jelmerk.knn.DistanceFunction;
import com.github.jelmerk.knn.DistanceFunctions;
import com.github.jelmerk.knn.Item;
import com.github.jelmerk.knn.SearchResult;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BinaryOperator;
import java.util.stream.Collectors;
import org.eclipse.collections.impl.list.mutable.FastList;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class EntryPointTest {

  static private final int maxItemCount = 29000;
  static private final int dimensions = 300;
  static private final int queryCount = 500;
  static private final String VECTOR_PATH = "";
  static private List<DTItem> items;
  static private List<double[]> queries;
  final Logger logger = LoggerFactory.getLogger(EntryPointTest.class);
  private final int m = 10;
  private final int efConstruction = 250;
  private final int ef = 20;
  private final int maxLevel = 4;
  private final DoubleDistanceFunctionWithCount distanceFunction = new DoubleDistanceFunctionWithCount(
      DistanceFunctions.DOUBLE_EUCLIDEAN_DISTANCE, 300);
  private final DoubleDistanceFunctionWithCount constrDistanceFunction = new DoubleDistanceFunctionWithCount(
      DistanceFunctions.DOUBLE_EUCLIDEAN_DISTANCE, 1);
  private final BinaryOperator<Double> differenceCalculator = DifferenceCalculators.DOUBLE_DIFFERENCE_CALCULATOR;
  List<SearchRecord> recordsHnsw = FastList.newList(100);
  List<SearchRecord> recordsSpatial = FastList.newList(100);
  List<SearchRecord> recordsEntryPoint = FastList.newList(100);
  private HnswIndexSpatial<Integer, double[], DTItem, Double> index;
  private HnswIndexSpatial2<Integer, double[], DTItem, Double> index2;

  @BeforeAll
  static void read() throws IOException {

    List<double[]> all = Files.lines(Paths.get(VECTOR_PATH))
                              .map(line -> Arrays.stream(line.split(",")).mapToDouble(Double::parseDouble).toArray())
                              .collect(Collectors.toList());

    Collections.shuffle(all);

    double minDist = Double.MAX_VALUE;
    double maxDist = 0;
    int sz = Math.min(all.size(), 1000);
    for (int i = 0; i < sz; i++) {
      for (int j = i + 1; j < sz; j++) {
        double dist = DistanceFunctions.DOUBLE_EUCLIDEAN_DISTANCE.distance(all.get(i), all.get(j));
        if (Arrays.equals(all.get(i), all.get(j))) {
          System.out.println(i + " and " + j + " are equal");
        }
        minDist = Math.min(minDist, dist);
        maxDist = Math.max(maxDist, dist);
      }
    }
    System.out.println("minDist: " + minDist);
    System.out.println("maxDist: " + maxDist);

    queries = all.subList(29000, 29500);
    AtomicInteger ai = new AtomicInteger();
    items = all.stream().limit(29000).map(vector -> new DTItem(ai.getAndIncrement(), vector)).collect(Collectors.toList());
    assertThat(queries.size(), is(queryCount));
    assertThat(items.size(), is(maxItemCount));
    System.out.println("Input ready.");
  }

  @Test
  void repeat() throws IOException, InterruptedException {
    final int repeat = 1;
    assert repeat <= queryCount;

    List<Integer> queryIndexes = new ArrayList<>();
    for (int i = 0; i < queryCount; i++) {
      queryIndexes.add(i);
    }
    Collections.shuffle(queryIndexes);

    for (int i = 0; i < repeat; i++) {
      logger.info("----- Test #{} -----", i);
      entryPointTest(queryIndexes.get(i));
    }
    System.out.println("No optimization:");
    for (SearchRecord searchRecord : recordsHnsw) {
      System.out.println(searchRecord);
    }
    System.out.println("Spatial (entry point unassigned):");
    for (SearchRecord searchRecord : recordsSpatial) {
      System.out.println(searchRecord);
    }
    System.out.println("Spatial (entry point assigned):");
    for (SearchRecord searchRecord : recordsEntryPoint) {
      System.out.println(searchRecord);
    }
  }

  void entryPointTest(int queryIndex) throws IOException, InterruptedException {
    final int K = 100;
    final int K1 = 10;

    double[] query = queries.get(queryIndex);
    List<Integer> entryPointIds = FastList.newList();

    System.out.println("======= No triangle optimization =======");
    logger.info("No triangle optim");
    recordsHnsw.add(new SearchRecord());
    List<SearchResult<DTItem, Double>> results0 = constructAndSearchHnsw(query, K);

    System.out.println("======= Default entry point =======");
    logger.info("default entry point");
    recordsSpatial.add(new SearchRecord());
    List<SearchResult<DTItem, Double>> results1 = constructAndSearch(entryPointIds, query, K, -1, false);

    // results1 is sorted in ascending order

    entryPointIds = results1.stream().skip(K - K1).map(result -> result.item().id()).collect(Collectors.toList());

    assert entryPointIds.size() == K1;

    int randomIndex = ThreadLocalRandom.current().nextInt(entryPointIds.size());
    int entryPointId = entryPointIds.get(randomIndex);
    System.out.println("======= Entry point #" + entryPointId + " =======");
    logger.info("entry point {}", entryPointId);
    recordsEntryPoint.add(new SearchRecord());
    List<SearchResult<DTItem, Double>> results2 = constructAndSearch(entryPointIds, query, K, entryPointId, false);

//    System.out.println("======= Exact =======");
//    List<SearchResult<DTItem, Double>> results3 = search(query, K, -1, true);

//    assertThat(results1, is(results0));
//    assertThat(results2, is(results1));
//    assertThat(results3, is(results2));
  }

  List<SearchResult<DTItem, Double>> constructAndSearchHnsw(double[] query, int k) throws InterruptedException {
    buildGraph2();
    return search2(query, k);
  }

  List<SearchResult<DTItem, Double>> constructAndSearch(List<Integer> entryPointIds, double[] query, int k,
                                                        int entryPointId, boolean exact)
      throws IOException, InterruptedException {
    buildGraph(entryPointIds);
    return search(query, k, entryPointId, exact);
  }

  void buildGraph2() throws InterruptedException {
    index2 = HnswIndexSpatial2.newBuilder(dimensions, distanceFunction, constrDistanceFunction, differenceCalculator,
                                          maxItemCount).withM(m).withEf(ef).withEfConstruction(efConstruction)
                              .withMaxLevel(maxLevel).<Integer, DTItem>refine()
                              .withEntryPointPicker((item) -> false)
                              .withEntryPointGetters((vector) -> null, (vector) -> null).build();
    Instant start = Instant.now();
    index2.addAll(items);
    Instant finish = Instant.now();
    long timeElapsed = Duration.between(start, finish).toMillis();
    System.out.println("------- index2.addAll(items) -------");
    SearchRecord record = recordsHnsw.get(recordsHnsw.size() - 1);
    record.constrTime = timeElapsed;
    record.constrConstrDistanceCount = constrDistanceFunction.getCount();
    record.constrDistanceCount = distanceFunction.getCount();
    System.out.printf("index construction time: %d ms\n", record.constrTime);
    System.out.printf("constr distance count: %d\n", record.constrConstrDistanceCount);
    System.out.printf("distance count: %d\n", record.constrDistanceCount);
  }

  void buildGraph(List<Integer> entryPointIds) throws IOException, InterruptedException {
    index = HnswIndexSpatial.newBuilder(dimensions, distanceFunction, constrDistanceFunction, differenceCalculator,
                                        maxItemCount).withM(m).withEf(ef).withEfConstruction(efConstruction)
                            .withMaxLevel(maxLevel).<Integer, DTItem>refine()
                            .withEntryPointPicker((item) -> entryPointIds.stream().anyMatch(
                                entryPointId -> entryPointId.equals(item.id())))
                            .withEntryPointGetters((vector) -> null, (vector) -> null).build();
    Instant start = Instant.now();
    index.addAll(items);
    Instant finish = Instant.now();
    long timeElapsed = Duration.between(start, finish).toMillis();
    System.out.println("------- index.addAll(items) -------");
    SearchRecord record;
    if (entryPointIds.size() == 0) {
      record = recordsSpatial.get(recordsSpatial.size() - 1);
    } else {
      record = recordsEntryPoint.get(recordsEntryPoint.size() - 1);
    }
    record.constrTime = timeElapsed;
    record.constrConstrDistanceCount = constrDistanceFunction.getCount();
    record.constrDistanceCount = distanceFunction.getCount();
    System.out.printf("index construction time: %d ms\n", record.constrTime);
    System.out.printf("constr distance count: %d\n", record.constrConstrDistanceCount);
    System.out.printf("distance count: %d\n", record.constrDistanceCount);
  }

  List<SearchResult<DTItem, Double>> search2(double[] query, int k) {
    Instant start = Instant.now();
    List<SearchResult<DTItem, Double>> results = index2.findNearest(query, k);
    Instant finish = Instant.now();
    long timeElapsed = Duration.between(start, finish).toMillis();
    System.out.println("------- indexHnsw.findNearest(query, k) -------");
    SearchRecord record = recordsHnsw.get(recordsHnsw.size() - 1);
    record.searchTime = timeElapsed;
    record.searchConstrDistanceCount = constrDistanceFunction.getCount();
    record.searchDistanceCount = distanceFunction.getCount();
    System.out.printf("search time: %d ms\n", record.searchTime);
    System.out.printf("constr distance count: %d\n", record.searchConstrDistanceCount);
    System.out.printf("distance count: %d\n", record.searchDistanceCount);
    return results;
  }

  List<SearchResult<DTItem, Double>> search(double[] query, int k, int entryPointId, boolean exact) {
    if (entryPointId < 0) {
      index.setEntryPointGetter((vector) -> null);  // Use fallback
    } else {
      index.setEntryPointGetter((vector) -> items.get(entryPointId));
    }

    List<SearchResult<DTItem, Double>> results;
    if (exact) {
      results = index.asExactIndex().findNearest(query, k);
    } else {
      // Repeat
      Instant start = Instant.now();
      results = index.findNearest(query, k);
      Instant finish = Instant.now();
      long timeElapsed = Duration.between(start, finish).toMillis();
      System.out.println("------- index.findNearest(query, k) -------");
      SearchRecord record;
      if (entryPointId < 0) {
        record = recordsSpatial.get(recordsSpatial.size() - 1);
      } else {
        record = recordsEntryPoint.get(recordsEntryPoint.size() - 1);
      }
      record.searchTime = timeElapsed;
      record.searchConstrDistanceCount = constrDistanceFunction.getCount();
      record.searchDistanceCount = distanceFunction.getCount();
      System.out.printf("search time: %d ms\n", record.searchTime);
      System.out.printf("constr distance count: %d\n", record.searchConstrDistanceCount);
      System.out.printf("distance count: %d\n", record.searchDistanceCount);
    }
    return results;
  }

  static class SearchRecord {

    long constrTime;
    int constrConstrDistanceCount;
    int constrDistanceCount;
    long searchTime;
    int searchConstrDistanceCount;
    int searchDistanceCount;

    public static SearchRecord getAverage(List<SearchRecord> list) {
      SearchRecord average = new SearchRecord();
      for (SearchRecord item : list) {
        average.constrTime += item.constrTime;
        average.constrConstrDistanceCount += item.constrConstrDistanceCount;
        average.constrDistanceCount += item.constrDistanceCount;
        average.searchTime += item.searchTime;
        average.searchConstrDistanceCount = item.searchConstrDistanceCount;
        average.searchDistanceCount = item.searchDistanceCount;
      }
      average.constrTime = Math.round((float) average.constrTime / list.size());
      average.constrConstrDistanceCount = Math.round((float) average.constrConstrDistanceCount / list.size());
      average.constrDistanceCount = Math.round((float) average.constrDistanceCount / list.size());
      average.searchTime = Math.round((float) average.searchTime / list.size());
      average.searchConstrDistanceCount = Math.round((float) average.searchConstrDistanceCount / list.size());
      average.searchDistanceCount = Math.round((float) average.searchDistanceCount / list.size());
      return average;
    }

    @Override
    public String toString() {
      return "" + constrTime
          + "\t" + constrConstrDistanceCount
          + "\t" + constrDistanceCount
          + "\t" + searchTime
          + "\t" + searchConstrDistanceCount
          + "\t" + searchDistanceCount;
    }
  }

  static class DTItem implements Item<Integer, double[]> {

    private static final long serialVersionUID = 1L;

    private final int id;
    private final double[] vector;
    private final long version;

    public DTItem(int id, double[] vector) {
      this(id, vector, 0);
    }

    public DTItem(int id, double[] vector, long version) {
      this.id = id;
      this.vector = vector;
      this.version = version;
    }

    @Override
    public Integer id() {
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
      return "DTItem{" + "id=" + id + "}";
    }
  }

  static class DoubleDistanceFunctionWithCount implements DistanceFunction<double[], Double> {

    private final DistanceFunction<double[], Double> distanceFunction;
    private final AtomicInteger count = new AtomicInteger();
    private final Integer repeat;

    public DoubleDistanceFunctionWithCount(DistanceFunction<double[], Double> distanceFunction, int repeat) {
      this.distanceFunction = distanceFunction;
      this.repeat = repeat;
    }

    @Override
    public Double distance(double[] u, double[] v) {
      count.getAndIncrement();

      double distance = 0.0;
      for (int i = 0; i < repeat; i++) {
        distance = this.distanceFunction.distance(u, v);
      }
      return distance;
    }

    public int getCount() {
      return count.getAndSet(0);
    }
  }
}
