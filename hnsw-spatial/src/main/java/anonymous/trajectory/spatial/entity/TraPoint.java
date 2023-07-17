package anonymous.trajectory.spatial.entity;

import java.io.Serializable;

/**
 * @author anonymous
 */
public class TraPoint implements Serializable {
    public double x, y;
    public long t, id;

    public TraPoint(double x, double y, long id) {
        this.x = x;
        this.y = y;
        this.id = id;
        this.t = 0;
    }

    public TraPoint(double x, double y, long t, long id) {
        this.x = x;
        this.y = y;
        this.t = t;
        this.id = id;
    }

    @Override
    public String toString() {
        return t + "," + x + "," + y + "\n";
    }
}
