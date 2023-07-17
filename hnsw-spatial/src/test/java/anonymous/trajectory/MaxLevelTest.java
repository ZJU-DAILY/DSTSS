package anonymous.trajectory;

import java.util.Arrays;
import org.junit.jupiter.api.Test;

// What max level is suitable?
public class MaxLevelTest {

  private final int m = 10;
  private final double levelLambda = 1 / Math.log(this.m);
  private final int count = 30000; // number of points

  @Test
  void test() {
    int[] arr = new int[10];
    for (int i = 0; i < count; i++) {
      int level = assignLevel(levelLambda);
      arr[level] += 1;
    }
    System.out.println(Arrays.toString(arr));
  }

  private int assignLevel(double lambda) {
    double random = Math.random();  // [0, 1]
    double r = -Math.log(random) * lambda;
    return (int) r;
  }
}
