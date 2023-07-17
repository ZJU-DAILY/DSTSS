package anonymous.trajectory.jobs;
import anonymous.trajectory.spatial.entity.Trajectory;
import anonymous.trajectory.spatial.measure.LCSS;
import anonymous.trajectory.spatial.measure.PointDistOnFlat;
import anonymous.trajectory.tools.DataReader;
import javafx.util.Pair;

import java.io.IOException;
import java.util.ArrayList;

/**
 * @author anonymous
 */
public class CalculateSim {
    public static void main(String[] args) throws IOException {

        ArrayList<Trajectory> trajectories = DataReader.readTexts(args[0], Long.parseLong(args[1]));
        ArrayList<Pair<Long, Double>> res = new ArrayList<>();
        LCSS lcss = new LCSS(new PointDistOnFlat(), Double.parseDouble(args[2]), Integer.parseInt(args[3]));
        for (Trajectory trajectory : trajectories) {
            System.out.println(trajectory.id());
            res.add(new Pair<>(trajectory.id(), lcss.traDist(trajectory, trajectories.get(Integer.parseInt(args[4])))));
        }
        res.sort((o1, o2) -> o2.getValue().compareTo(o1.getValue()));
        System.out.println(trajectories.get(Integer.parseInt(args[4])).toString());
        System.out.println("===================================");
        for (int i = 0; i < Integer.parseInt(args[5]); i++) {
            System.out.println(trajectories.get(res.get(i).getKey().intValue()).vector().size());
            System.out.println(res.get(i).getValue());
            System.out.println("******************************************");
        }
    }
}
