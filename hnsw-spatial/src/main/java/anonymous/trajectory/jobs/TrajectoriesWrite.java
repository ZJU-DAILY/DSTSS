package anonymous.trajectory.jobs;

import anonymous.trajectory.spatial.entity.Trajectory;
import anonymous.trajectory.tools.DataReader;

import java.io.*;
import java.util.List;

/**
 * @author anonymous
 */
public class TrajectoriesWrite {
    public static void main(String[] args) throws IOException {
        List<Trajectory> trajectories = DataReader.readTexts(
                "", 1800000);
        System.out.println(trajectories.size());
        //将每次计算的结果写入到文件最后一行，而不是覆盖
        File file = new File("");
        for (Trajectory trajectory : trajectories) {
            if (trajectory.id > 1000) {
                return;
            }
            if (trajectory.id % 1000 == 0) {
                System.out.println(trajectory.id);
            }
            try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file,true)))){
                bw.write(trajectory.toString());
            }catch(Exception e) {
                e.printStackTrace();
            }
    }


//        for (Trajectory trajectory : trajectories) {
//            System.out.println(trajectory);
//        }
    }
}
