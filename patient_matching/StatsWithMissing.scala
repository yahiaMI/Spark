/**
 * Note that instead of calling the map function to generate an Array[NAStatCounter] for
 * each record in the input RDD, we’re calling the slightly more advanced mapParti
 * tions function, which allows us to process all of the records within a partition of the
 * input RDD[Array[Double]] via an Iterator[Array[Double]].
 *
 * This allows us to create a single instance of Array[NAStatCounter] for each partition of the data and then update its state using the Array[Double] values that are returned by the given iterator,which is a more efficient implementation.
 */

import org.apache.spark.rdd.RDD
def statsWithMissing(rdd: RDD[Array[Double]]): Array[NAStatCounter] = {

  val nastats = rdd.mapPartitions((iter: Iterator[Array[Double]]) => {

    val nas: Array[NAStatCounter] = iter.next().map(d => NAStatCounter(d))

    iter.foreach(arr => {

      nas.zip(arr).foreach { case (n, d) => n.add(d) }

    })

    Iterator(nas)

  })

  nastats.reduce((n1, n2) => {

    n1.zip(n2).map { case (a, b) => a.merge(b) }

  })
}
