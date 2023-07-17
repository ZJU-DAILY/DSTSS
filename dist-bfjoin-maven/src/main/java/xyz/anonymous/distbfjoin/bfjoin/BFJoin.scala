package xyz.anonymous.distbfjoin.bfjoin

import java.util
import xyz.anonymous.distbfjoin.LocationDistanceFunctions.EUCLIDEAN_DISTANCE_FUNCTION
import xyz.anonymous.distbfjoin.TextDistanceFunctions.JACCARD_SIMILARITY
import xyz.anonymous.distbfjoin.utils.DoubleIdentical.DOUBLE_IDENTICAL
import org.apache.spark.HashPartitioner
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import xyz.anonymous.distbfjoin.{EnvUtil, ExactTrajectorySimilarity, Result, Trajectory, UpperBoundTrajectorySimilarity}
import xyz.anonymous.distbfjoin.utils.{Batch, DoubleSigmoid}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.collection.Map

/** The BFJoin index.
 *
 * @param alpha     the relative importance of location similarity to text similarity
 * @param gamma     parameter for the Sigmoid function
 * @param diagTheta threshold for MBR diagonal length
 */
class BFJoin(private val alpha: Double,
             private val gamma: Double,
             private val diagTheta: Double,
             private val numPartitions: Int,
             private val kwAndBatches: RDD[(Int, ArrayBuffer[Batch])]
            ) {

  // kwAndBatches.persist(StorageLevel.MEMORY_AND_DISK_SER)

  private val normalizer = new DoubleSigmoid(gamma)
  private val uts = new UpperBoundTrajectorySimilarity(alpha, normalizer, DOUBLE_IDENTICAL)
  private val ts = new ExactTrajectorySimilarity(
    alpha,
    EUCLIDEAN_DISTANCE_FUNCTION,
    JACCARD_SIMILARITY,
    normalizer,
    DOUBLE_IDENTICAL
  )

  def search(queries: RDD[Trajectory], k: Int): Map[Int, (List[Result], Long)] = {
    val task = new BFJoinTask(alpha, uts, ts, normalizer)
    val kwAndQueries = Batchifier.group(queries)
    val joined = numPartitions match {
      case 0 => kwAndBatches.join(kwAndQueries, new HashPartitioner(EnvUtil.ctx.defaultParallelism))
      case x if x > 0 => kwAndBatches.join(kwAndQueries, new HashPartitioner(x))
      case _ => kwAndBatches.join(kwAndQueries)
    }
    joined.flatMap(pair => {
      val kw = pair._1
      val batches = pair._2._1.asJava
      val queries = pair._2._2
      queries.map(query => (query.getId, task.call(query, batches, k)))
    }).reduceByKey((q1, q2) => {
      while (!q2.getResults.isEmpty) {
        val r = q2.getResults.poll()
        if (!q1.getResults.contains(r)) {
          q1.getResults.add(r)
        }
        if (q1.getResults.size > k) {
          q1.getResults.poll()
        }
      }
      new TaskResult(q1.getResults, q1.getTime + q2.getTime, q1.getCount + q2.getCount)
    }).mapValues(pq => {
      var arr = List[Result]()
      while (!pq.getResults.isEmpty) {
        arr = pq.getResults.poll() +: arr
      }
      (arr, pq.getTime / pq.getCount)
    }).collectAsMap()
  }
}
