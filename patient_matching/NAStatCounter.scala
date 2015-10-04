/**
 * NAStatCounter is an StatCounter improvment.
 * We get the statistics of a set of numbers (count, mean and variance) and Nan number.
 */

import org.apache.spark.util.StatCounter

class NAStatCounter extends Serializable {

  val stats: StatCounter = new StatCounter()
  var missing: Long = 0

  // if the value is missing, we increment the number of missing
  // else we merge the NAStatCounter
  def add(x: Double): NAStatCounter = {
    if (java.lang.Double.isNaN(x)) {
      missing += 1
    } else {
      stats.merge(x)
    }
    this
  }

  def merge(other: NAStatCounter): NAStatCounter = {
    stats.merge(other.stats)
    missing += other.missing
    this
  }

  // we override the toString function
  override def toString = {
    "stats: " + stats.toString + " NaN: " + missing
  }
}

object NAStatCounter extends Serializable {
  def apply(x: Double) = new NAStatCounter().add(x)
}

