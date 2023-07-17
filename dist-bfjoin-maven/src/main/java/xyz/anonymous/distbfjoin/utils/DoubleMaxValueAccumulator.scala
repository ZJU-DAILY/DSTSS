package xyz.anonymous.distbfjoin.utils

import java.{lang => jl}
import org.apache.spark.util.AccumulatorV2

/**
 * An [[AccumulatorV2 accumulator]] for recording the max value of a series of double precision
 * floating numbers.
 */
class DoubleMaxValueAccumulator extends AccumulatorV2[jl.Double, jl.Double] {

  private var _max = 0.0

  override def isZero: Boolean = _max == 0.0

  override def copy(): DoubleMaxValueAccumulator = {
    val newAcc = new DoubleMaxValueAccumulator
    newAcc._max = _max
    newAcc
  }

  override def reset(): Unit = {
    _max = 0.0
  }

  override def add(v: jl.Double): Unit = {
    if (v > _max) {
      _max = v
    }
  }

  def max: Double = _max

  override def merge(other: AccumulatorV2[jl.Double, jl.Double]): Unit = other match {
    case o: DoubleMaxValueAccumulator =>
      _max = Math.max(_max, o.max)
    case _ =>
      throw new UnsupportedOperationException(
        s"Cannot merge ${this.getClass.getName} with ${other.getClass.getName}")
  }

  override def value: jl.Double = _max
}
