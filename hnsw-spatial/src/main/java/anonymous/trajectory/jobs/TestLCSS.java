package anonymous.trajectory.jobs;

import anonymous.trajectory.spatial.entity.Trajectory;
import anonymous.trajectory.spatial.measure.LCSS;
import anonymous.trajectory.spatial.measure.PointDistOnFlat;
import anonymous.trajectory.tools.DataReader;
import javafx.util.Pair;

import java.io.IOException;
import java.util.ArrayList;

public class TestLCSS {
    public static void main(String[] args) throws IOException {
        ArrayList<Trajectory> trajectories = DataReader.readTexts(
                "", 1800000);
        ArrayList<Pair<Long, Double>> res = new ArrayList<>();
        LCSS lcss = new LCSS(new PointDistOnFlat(), 2000, 5);
        System.out.println(trajectories.get(0).toString());
        System.out.println(trajectories.get(50008).toString());
        System.out.println(lcss.traDist(trajectories.get(0), trajectories.get(50008)));
    }
}
