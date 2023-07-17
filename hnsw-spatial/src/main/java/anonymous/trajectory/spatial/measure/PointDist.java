package anonymous.trajectory.spatial.measure;

import anonymous.trajectory.spatial.entity.TraPoint;

/**
 * @author anonymous
 */
public interface PointDist {
    /**
     * 计算两点之间的距离
     * @param p1 点1
     * @param p2 点2
     * @return 两点距离
     */
    double calc(TraPoint p1, TraPoint p2);
}
