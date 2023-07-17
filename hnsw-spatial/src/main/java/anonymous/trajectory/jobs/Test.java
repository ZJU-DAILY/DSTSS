package anonymous.trajectory.jobs;

import anonymous.trajectory.spatial.entity.Trajectory;
import anonymous.trajectory.spatial.measure.LCSS;
import anonymous.trajectory.spatial.measure.PointDistOnGPS;
import anonymous.trajectory.tools.DataReader;
import javafx.util.Pair;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;

public class Test {
    public static void main(String[] args) throws IOException {
        ArrayList<Trajectory> trajectories = DataReader.readTexts(
                "", 1800000);
        ArrayList<Pair<Long, Double>> res = new ArrayList<>();
        LCSS lcss = new LCSS(new PointDistOnGPS(), 2000, 5);
        for (Trajectory trajectory : trajectories) {
            System.out.println(trajectory.id());
            res.add(new Pair<>(trajectory.id(), lcss.traDist(trajectory, trajectories.get(50008))));
        }
        res.sort(Comparator.comparing(Pair::getValue));
        System.out.println(trajectories.get(50008).toString());
        System.out.println("===================================");
        for (int i = 0; i < 10; i++) {
            System.out.println(trajectories.get(res.get(i).getKey().intValue()).toString());
            System.out.println(res.get(i).getValue());
            System.out.println("******************************************");
        }
    }
}
