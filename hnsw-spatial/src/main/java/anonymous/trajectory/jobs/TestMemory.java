package anonymous.trajectory.jobs;

import java.util.ArrayList;
import java.util.Random;

/**
 * @author anonymous
 */
public class TestMemory {
    public static void main(String[] args) {
        Runtime r = Runtime.getRuntime();
        Random rand=new Random();
        long startMemory = r.freeMemory();
        ArrayList<Integer> list = new ArrayList<>();
        for (int i = 0; i < 200000; i++) {
            list.add(rand.nextInt());
        }
        long endMemory = r.freeMemory();
        System.out.println(startMemory - endMemory);
        System.out.println(list.size());
    }
}
