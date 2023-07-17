package xyz.anonymous.distbfjoin

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

object EnvUtil {

  private val session = new ThreadLocal[SparkSession]()

  def set(sparkSession: SparkSession): Unit = {
    session.set(sparkSession);
  }

  def get: SparkSession = {
    session.get
  }

  def remove(): Unit = {
    session.remove()
  }

  def ctx: SparkContext = {
    session.get.sparkContext
  }
}
