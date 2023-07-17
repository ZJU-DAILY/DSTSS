package anonymous.trajectory.tools;


import anonymous.trajectory.spatial.entity.TraPoint;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * @author anonymous
 */
public class DataTransformer {

    public static long getStringToDate(String time, String format) {
        // "yyyy-MM-dd HH:mm:ss"
        SimpleDateFormat sf = new SimpleDateFormat(format);
        Date date = new Date();
        try {
            date = sf.parse(time);
        } catch (ParseException e) {
            e.printStackTrace();
        }
        return date.getTime();
    }
    public static TraPoint line2TraPoint (String line) {
        long id = Long.parseLong(line.split(",")[0]);
        double x = Double.parseDouble(line.split(",")[2]);
        double y = Double.parseDouble(line.split(",")[3]);
        long t = DataTransformer.getStringToDate(line.split(",")[1], "yyyy-MM-dd HH:mm:ss");

        return new TraPoint(x, y, t, id);
    }


}
