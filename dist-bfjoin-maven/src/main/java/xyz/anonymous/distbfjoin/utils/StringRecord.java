package xyz.anonymous.distbfjoin.utils;

import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

public class StringRecord {

  private final Map<String, Integer> map = new Object2IntOpenHashMap<>(4000);
  private int count = 0;

  public void preread(String path) {
    try {
      List<String> keywords = Files.readAllLines(Paths.get(path));
      for (String keyword : keywords) {
        map.put(keyword.trim(), count);
        count++;
      }
    } catch (Exception ignored) {

    }
  }

  public int get(String str) {
    if (!map.containsKey(str))  {
      synchronized (this) {
        if (!map.containsKey(str)) {
          map.put(str, count);
          count++;
        }
      }
    }

    return map.get(str);
  }
}
