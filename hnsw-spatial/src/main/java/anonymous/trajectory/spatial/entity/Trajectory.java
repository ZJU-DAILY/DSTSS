package anonymous.trajectory.spatial.entity;

import com.github.jelmerk.knn.Item;

import java.util.LinkedList;

/**
 * @author anonymous
 */
public class Trajectory implements Item<Long, LinkedList<TraPoint>> {

    public long id;
    public LinkedList<TraPoint> points;

    public Trajectory() {
        this.id = -1;
        this.points = new LinkedList<>();
    }

    public void
    addPoint(TraPoint p, long id) {
        if (points.isEmpty()) {
            this.id = id;
        }
        points.addLast(p);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        for (TraPoint p : points) {
            builder.append(id).append(",").append(p.toString());
        }
        return builder.toString();
    }

    @Override
    public Long id() {
        return this.id;
    }

    @Override
    public LinkedList<TraPoint> vector() {
        return this.points;
    }

    @Override
    public int dimensions() {
        return 100;
    }
}
