package anonymous.trajectory.spatial.measure;

import anonymous.trajectory.spatial.entity.Trajectory;

/**
 * @author anonymous
 */
public interface Measure {
    /**
     * 计算两条轨迹的相似性
     * @param t1 轨迹1
     * @param t2 轨迹2
     * @return 两条轨迹的相似性
     */
    double traDist(Trajectory t1, Trajectory t2);
}
