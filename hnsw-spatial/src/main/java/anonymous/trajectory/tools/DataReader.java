package anonymous.trajectory.tools;

import anonymous.trajectory.spatial.entity.TraPoint;
import anonymous.trajectory.spatial.entity.Trajectory;
import anonymous.trajectory.spatial.measure.PointDistOnGPS;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.stream.Stream;

/**
 * @author anonymous
 */
public class DataReader {


    public static ArrayList<Trajectory> readTexts(String folderName, long timeThreshold) throws IOException {

        ArrayList<Trajectory> result = new ArrayList<Trajectory>();
        long traId = 0;
        PointDistOnGPS dis = new PointDistOnGPS();
        try (Stream<Path> paths = Files.walk(Paths.get(folderName))) {
            paths
                    .filter(Files::isRegularFile)
                    .forEach(x -> {
                        TraPoint lastPoint = null;
                        File y = x.toFile();
                        InputStreamReader reader;
                        try {
                            reader = new InputStreamReader(new FileInputStream(y), StandardCharsets.UTF_8);
                        } catch (FileNotFoundException e) {
                            throw new RuntimeException(e);
                        }
                        BufferedReader bufferedReader = new BufferedReader(reader);
                        String line;

                        while (true) {
                            try {
                                if ((line = bufferedReader.readLine()) == null) {
                                    break;
                                }
                                else {
                                    TraPoint point = DataTransformer.line2TraPoint(line);
                                    if (lastPoint == null || point.t - lastPoint.t >= timeThreshold ||
                                            (result.size() > 0 && result.get(result.size() - 1).vector().size() >= 50)) {

                                        if (result.size() > 0 && (result.get(result.size() - 1).vector().size() < 30 ||
                                                dis.calc(result.get(result.size() - 1).vector().get(0),
                                                        result.get(result.size() - 1).vector().get(result.get(result.size() - 1).vector().size() - 1)) < 2000)) {
                                            result.remove(result.size() - 1);
                                        }
                                        Trajectory newTrajectory = new Trajectory();
                                        newTrajectory.addPoint(point, result.size());


                                        result.add(newTrajectory);
                                    }
                                    else {
                                        if (point.x != lastPoint.x && point.y != lastPoint.y) {
                                            result.get(result.size() - 1).addPoint(point, result.size() - 1);
                                        }
                                    }
                                    lastPoint = point;
                                }
                            } catch (IOException e) {
                                throw new RuntimeException(e);
                            }

                        }

                        try {
                            reader.close();
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    });
        }

        return result;


    }
}
