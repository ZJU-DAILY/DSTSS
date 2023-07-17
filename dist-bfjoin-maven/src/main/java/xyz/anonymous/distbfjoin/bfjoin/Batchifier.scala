package xyz.anonymous.distbfjoin.bfjoin

import org.apache.spark.HashPartitioner

import scala.collection.convert.ImplicitConversions.`iterable AsScalaIterable`
import org.apache.spark.rdd.RDD
import org.apache.spark.util.SizeEstimator
import xyz.anonymous.distbfjoin.utils.Batch
import xyz.anonymous.distbfjoin.{EnvUtil, Trajectory}

import java.util
import scala.collection.mutable.ArrayBuffer

object Batchifier {

  def group(rdd: RDD[Trajectory], numPartitions: Int = 0): RDD[(Int, Iterable[Trajectory])] = {
    // to inverted lists
    val groups = rdd.flatMap(t => {
      val pairs = ArrayBuffer[(Int, Trajectory)]()
      for (kw <- t.getTermIterator) {
        pairs += ((kw.toInt, t))
      }
      pairs
    })
    numPartitions match {
      case 0 => groups.groupByKey(new HashPartitioner(EnvUtil.ctx.defaultParallelism))
      case x if x > 0 => groups.groupByKey(new HashPartitioner(numPartitions))
      case _ => groups.groupByKey()
    }
  }

  def batchify(rdd: RDD[Trajectory], diagTheta: Double, numPartitions: Int = 0, debug: Boolean =
  false): RDD[(Int, ArrayBuffer[Batch])] = {
    // batchify
    val groups = group(rdd, numPartitions)

    if (debug) {
      println(groups.countByKey())
    }

    groups.mapValues(trs => {
      val batches = ArrayBuffer[Batch]()
      for (t <- trs) {
        var bestBatch: Batch = null
        var minDiag = Double.MaxValue
        var minMbr: MBR = null
        val trajectoryMbr = t.summarize.getMbr
        for (batch <- batches) {
          val newMbr: MBR = batch.getMbr.include(trajectoryMbr)
          val newDiag = newMbr.getDiag
          if (newDiag < minDiag) {
            minDiag = newDiag
            minMbr = newMbr
            bestBatch = batch
          }
        }
        if (bestBatch == null || minDiag > diagTheta) {
          bestBatch = new Batch()
          batches += bestBatch
          minMbr = trajectoryMbr
        }
        bestBatch.add(t, minMbr)
      }
      if (debug) {
        println(s"size of the batch: ${SizeEstimator.estimate(batches)}")
      }
      batches
    })
  }
}
