package anonymous.trajectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class DataSampler {

  private final List<double[]> arrays;

  public DataSampler(String path) throws IOException {
    Path path1 = Paths.get(path);
    this.arrays = Files.lines(path1)
        .map(line -> Arrays.stream(line.split(" ")).skip(1).mapToDouble(Double::parseDouble).toArray())
        .collect(Collectors.toList());
  }

  public List<double[]> sample(int count) {
    assert count == 30000;
    return arrays;
  }
}
