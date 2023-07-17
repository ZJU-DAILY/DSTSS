package xyz.anonymous.distbfjoin.dao

import org.apache.spark.rdd.RDD
import xyz.anonymous.distbfjoin.EnvUtil

object TextDao {

  def textRDD(filename: String): RDD[(String, Long)] = {
    EnvUtil.ctx.textFile(filename).zipWithIndex()
  }
}
