package xyz.anonymous.distbfjoin.dao

import org.apache.spark.rdd.RDD
import xyz.anonymous.distbfjoin.{Point, Trajectory}

object TrajectoryDao {

  /**
   * Read trajectory RDDs from the text file.
   *
   * @param path          path to the text file
   * @param minPartitions suggested minimum number of partitions
   * @param gridSize      grid size
   * @param fraction      sample rate (.0 means do not sample)
   * @param hasTimestamp  whether the dataset has timestamps
   * @return trajectory RDDs
   */
  def fromText(path: String, gridSize: Double, fraction: Double = 0, hasTimestamp: Boolean =
  false, parallelism: Int = 0): RDD[Trajectory] = {
    val rdd = TextDao.textRDD(path).map(pair => {
      val fields = pair._1.split("\t")
      val lineNo = pair._2
      val tid = fields(0).toInt
      val x = if (hasTimestamp) fields(2).toDouble else fields(1).toDouble
      val y = if (hasTimestamp) fields(3).toDouble else fields(2).toDouble
      val kwIdx = if (hasTimestamp) 4 else 3
      val kwSet = if (fields.length > kwIdx) fields(kwIdx).split(",") else null
      val p = new Point(lineNo, x, y, kwSet)
      (tid, p)
    })

    val pointToTrajectory = (p: Point) => {
      val t = new Trajectory(gridSize)
      t.addPoint(p)
      t
    }
    val addPointToTrajectory = (t: Trajectory, p: Point) => {
      t.addPoint(p)
      t
    }
    val mergeTrajectories = (t1: Trajectory, t2: Trajectory) => {
      t1.addPoints(t2.getPoints)
      t1
    }
    val trajectories = rdd.combineByKey(pointToTrajectory, addPointToTrajectory, mergeTrajectories)
        .map(pair => {
          val tid = pair._1
          val t = pair._2
          t.setId(tid).sort()
        })
    if (fraction > 0 && fraction < 1) {
      trajectories.sample(false, fraction)
    } else if (fraction > 1) {
      trajectories.flatMap(trajectory => (1 to fraction.toInt).map(_ => trajectory))
    } else {
      trajectories
    }
  }
}
