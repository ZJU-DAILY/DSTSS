package anonymous.trajectory;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.CoreMatchers.is;

import com.github.jelmerk.knn.DistanceFunction;
import com.github.jelmerk.knn.DistanceFunctions;
import com.github.jelmerk.knn.JavaObjectSerializer;
import com.github.jelmerk.knn.ObjectSerializer;
import com.github.jelmerk.knn.SearchResult;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.function.BinaryOperator;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

public class HnswIndexSpatialToyTest {
  private final int maxItemCount = 100;
  private final int m = 12;
  private final int efConstruction = 250;
  private final int ef = 20;
  private final int dimensions = 2;
  private final int maxLevel = 3;
  private final DistanceFunction<float[], Float> distanceFunction = DistanceFunctions.FLOAT_EUCLIDEAN_DISTANCE;
  private final DistanceFunction<float[], Float> constrDistanceFunction = DistanceFunctions.FLOAT_EUCLIDEAN_DISTANCE;
  private final BinaryOperator<Float> differenceCalculator = DifferenceCalculators.FLOAT_DIFFERENCE_CALCULATOR;
  private final ObjectSerializer<String> itemIdSerializer = new JavaObjectSerializer<>();
  private final ObjectSerializer<TestItem> itemSerializer = new JavaObjectSerializer<>();
  private final TestItem item1 = new TestItem("1", new float[]{0.0110f, 0.2341f}, 10);
  private final TestItem item2 = new TestItem("2", new float[]{0.2300f, 0.3891f}, 10);
  private final TestItem item3 = new TestItem("3", new float[]{0.4300f, 0.9891f}, 10);
  private final TestItem item4 = new TestItem("4", new float[]{0.9999f, 0.9999f}, 10);
  private HnswIndexSpatial2<String, float[], TestItem, Float> index;

  void setUp() throws IOException {
    index = HnswIndexSpatial2
        .newBuilder(dimensions, distanceFunction, constrDistanceFunction, differenceCalculator, maxItemCount)
        .withCustomSerializers(itemIdSerializer, itemSerializer)
        .withM(m)
        .withEfConstruction(efConstruction)
        .withEf(ef)
        .withRemoveEnabled()
        .withMaxLevel(maxLevel)
        .withEntryPointPicker((item) -> item == item2)
        .withEntryPointGetters((vector) -> item2, (vector) -> item2)
        .build();
  }

  void setUp2() throws IOException {
    index = HnswIndexSpatial2
        .newBuilder(dimensions, distanceFunction, constrDistanceFunction, differenceCalculator, maxItemCount)
        .withCustomSerializers(itemIdSerializer, itemSerializer)
        .withM(m)
        .withEfConstruction(efConstruction)
        .withEf(ef)
        .withRemoveEnabled()
        .withMaxLevel(maxLevel)
        .withEntryPointPicker((item) -> item == item2 || item == item3)
        .withEntryPointGetters((vector) -> item2, (vector) -> item3)
        .build();
  }

  @AfterEach
  void tearDown() {
    System.out.println(index);
  }

  @Test
  void returnMaxLevel() throws IOException {
    setUp();
    assertThat(index.getMaxLevel(), is(maxLevel));
  }

  @Test
  void addAll() throws InterruptedException, IOException {
    setUp();
    index.addAll(Arrays.asList(item1, item2, item3));
    assertThat(index.isEntryPoint(item1.id()), is(false));
    assertThat(index.isEntryPoint(item2.id()), is(true));
    assertThat(index.isEntryPoint(item3.id()), is(false));
  }

  @Test
  void search() throws InterruptedException, IOException {
    setUp2();
    index.addAll(Arrays.asList(item1, item2, item3, item4));
    List<SearchResult<TestItem, Float>> result = index.findNearest(item2.vector(), 10);
    assertThat(result, is(Arrays.asList(
        SearchResult.create(item2, 0f),
        SearchResult.create(item1, 0.26830205f),
        SearchResult.create(item3, 0.6324555f),
        SearchResult.create(item4, 0.98276275f)
    )));
  }
}
