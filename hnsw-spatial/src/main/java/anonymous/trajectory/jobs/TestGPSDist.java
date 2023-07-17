package anonymous.trajectory.jobs;

import anonymous.trajectory.spatial.entity.TraPoint;
import anonymous.trajectory.spatial.measure.PointDistOnGPS;

/**
 * @author anonymous
 */
public class TestGPSDist
{
    public static void main(String[] args) {
        PointDistOnGPS dis = new PointDistOnGPS();
        TraPoint p1 = new TraPoint(116.74599,40.16957, 1);
        TraPoint p2 = new TraPoint(116.37497,39.85789, 2);
        System.out.println(dis.calc(p1, p2));
    }
}
