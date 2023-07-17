package xyz.anonymous.distbfjoin;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class TextDistanceFunctions {

  public static class JaccardSimilarity implements DistanceFunction<int[]> {

    JaccardSimilarity() {
    }

    /**
     * Return the distance between two keyword sets. Keywords are represented by their numeric
     * indexes.
     *
     * @param a keyword set a
     * @param b keyword set b
     * @return the distance within [0, 1], the bigger, the more similar
     */
    @Override
    public double distance(int[] a, int[] b) {
      List<Integer> aList = new IntArrayList(a);
      List<Integer> bList = new IntArrayList(b);

      Set<Integer> union = new IntOpenHashSet(aList);
      union.addAll(bList);
      if (union.size() == 0) {
        return 1;
      }
      Set<Integer> intersection = new IntOpenHashSet(aList);
      intersection.retainAll(bList);
      return ((double) intersection.size()) / union.size();
    }
  }

  public static JaccardSimilarity JACCARD_SIMILARITY = new JaccardSimilarity();
}
