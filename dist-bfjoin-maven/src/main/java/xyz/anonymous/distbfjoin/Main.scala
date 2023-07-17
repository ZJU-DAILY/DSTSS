package xyz.anonymous.distbfjoin

import LocationDistanceFunctions.EuclideanDistanceFunction
import org.apache.hadoop.thirdparty.com.google.common.hash.BloomFilter
import org.apache.hadoop.thirdparty.com.google.common.hash.BloomFilterStrategies
import it.unimi.dsi.fastutil.ints.{Int2ObjectOpenHashMap, IntOpenHashSet}
import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet
import org.apache.logging.log4j.LogManager
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.SizeEstimator
import xyz.anonymous.distbfjoin.bfjoin.{BFJoin, BFJoinTask, Batchifier, MBR, TaskResult}
import xyz.anonymous.distbfjoin.dao.TrajectoryDao
import xyz.anonymous.distbfjoin.utils.{Batch, DoubleIdentical, DoubleSigmoid, Timer}

import scala.collection.JavaConverters._
import java.{lang => jl, util => ju}
import java.util.Random
import scala.annotation.tailrec
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.StdIn.readLine

object Main {

  private val VERSION = "01201608"
  private val PATH = "/user/anonymous/augmented/brinkhoff-sample-0.4-50x-no-fill"

  def main(args: Array[String]): Unit = {
    println("Ver: " + VERSION)

    // args (default brinkhoff)
    val path = args(0)
    val queryPath = args(1)

    println(s"path = ${path}")
    println(s"queryPath = ${queryPath}")
    val params =
      if (args.length > 0) getParams(args, 2) else new ju
      .HashMap[String, String]()
    println(s"Params: $params")
    val alpha = params.getOrDefault("alpha", "0.5").toDouble
    val gamma = params.getOrDefault("gamma", "0.00001").toDouble
    val diagTheta = params.getOrDefault("diag-theta", "130000").toDouble
    val gridSize = params.getOrDefault("grid-size", "26000").toDouble
    val numPartitions = params.getOrDefault("num-partition", "0").toInt
    val k = params.getOrDefault("k", "50").toInt
    val hasTimestamp = !params.containsKey("no-timestamp")
    val useJavaSer = params.containsKey("use-java-ser")
    val numQueries = params.getOrDefault("num-queries", "1").toInt
    val debug = params.containsKey("debug")
    val fraction = params.getOrDefault("fraction", "0").toDouble

    // create spark context
    val conf = new SparkConf().setAppName("dist-bfjoin")
    if (!useJavaSer) {
      conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      conf.set("spark.kryo.registrationRequired", "true")
      conf.set("spark.kryoserializer.buffer.max", "1024m")
      conf.registerKryoClasses(Array[Class[_]](
        classOf[Trajectory],
        classOf[Result],
        classOf[Point],
        classOf[MBR],
        classOf[ju.ArrayList[_]],
        classOf[ju.HashSet[_]],
        classOf[ju.PriorityQueue[_]],
        classOf[Batch],
        classOf[BloomFilter[_]],
        classOf[BFJoinTask],
        classOf[DoubleSigmoid],
        classOf[EuclideanDistanceFunction],
        classOf[TextDistanceFunctions.JaccardSimilarity],
        classOf[UpperBoundTrajectorySimilarity],
        classOf[ExactTrajectorySimilarity],
        classOf[DoubleIdentical],
        classOf[ju.concurrent.atomic.AtomicLongArray],
        classOf[java.nio.charset.Charset],
        classOf[Int2ObjectOpenHashMap[_]],
        classOf[IntOpenHashSet],
        classOf[ObjectOpenHashSet[_]],
        classOf[TaskResult],
        Class.forName("org.apache.hadoop.thirdparty.com.google.common.hash" +
            ".BloomFilterStrategies$LockFreeBitArray"),
        Class.forName("org.apache.hadoop.thirdparty.com.google.common.hash.LongAdder"),
        Class.forName("org.apache.hadoop.thirdparty.com.google.common.hash" +
            ".Funnels$StringCharsetFunnel"),
        Class.forName("org.apache.hadoop.thirdparty.com.google.common.hash.BloomFilterStrategies"),
        Class.forName("sun.nio.cs.UTF_8"),
        Class.forName("[[B"),
        Class.forName("scala.reflect.ClassTag$GenericClassTag"),
        Class.forName("scala.math.LowPriorityOrderingImplicits$$anon$2"),
        Class.forName("scala.Predef$$anon$2"),
        Class.forName("[Lxyz.anonymous.distbfjoin.Trajectory;")
      ))
    }
    val spark = SparkSession.builder.config(conf).getOrCreate()
    EnvUtil.set(spark)
    // EnvUtil.ctx.addJar("/home/anonymous/dist-bfjoin-maven/target/dist-bfjoin-1.0-SNAPSHOT.jar")
    println("App: " + EnvUtil.ctx.appName)

    // create index
    var timer = new Timer()
    val trs = TrajectoryDao.fromText(path, gridSize, fraction, hasTimestamp).cache()
    val trCount = trs.count.toInt
    println(s"Trajectory count: ${trCount}")
    val ptCount = trs.map(t => t.length()).sum().toInt
    println(s"Point count: ${ptCount}")

    val kwAndBatches = Batchifier.batchify(trs, diagTheta, numPartitions, debug = debug)

    val index = new BFJoin(alpha, gamma, diagTheta, numPartitions, kwAndBatches)
    val kwCount = kwAndBatches.count
    println(s"Index created in ${timer.duration} ms")
    println(s"Keyword count: ${kwCount}")

    val random = new Random(48)

    val queries =
      if (queryPath.equals("sample")) {
        val indexes = ArrayBuffer[Int]()
        for (_ <- 0 until numQueries) {
          indexes += random.nextInt(trCount)
        }
        trs.filter(t => indexes.contains(t.getId))
      } else {
        TrajectoryDao.fromText(queryPath, gridSize, numQueries, hasTimestamp)
      }

    val queriesCount = queries.count.toInt
    println(s"${queriesCount} queries in total.")

    timer = new Timer()
    val queriesAndResults = index.search(queries, k)
    val queryTime = timer.duration
    println("Queries completed in " + queryTime + " ms in total, average time = " +
        (queryTime.toFloat / queriesCount) + " ms")

    for ((kw, (results, time)) <- queriesAndResults) {
      println(s"Query: trajectory #$kw completed in $time ms")
      val builder = new mutable.StringBuilder()
      results.foreach(result => {
        print(f"${result.trajectory.getId}%7d ")
      })
      println()
      builder.clear()
      results.foreach(result => {
        print(f"${result.similarity}%.5f ")
      })
      println()
      println()
    }
  }

  private def getParams(args: Array[String], startIdx: Int = 0): ju.Map[String, String] = {
    val params = new ju.HashMap[String, String]()
    val it = args.iterator
    var key: String = null

    for (_ <- 0 until startIdx) {
      it.next()
    }
    while (it.hasNext) {
      val next = it.next()
      if (next.startsWith("--")) {
        if (key != null) {
          params.put(key, "true")
        }
        key = next.substring(2)
      } else {
        if (key != null) {
          params.put(key, next)
          key = null
        } else {
          throw new IllegalArgumentException(next)
        }
      }
    }
    if (key != null) {
      params.put(key, "true")
    }
    params
  }
}
