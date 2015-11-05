
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

class ExtractTransform {

  def getData(sc: SparkContext, hdfs_path: String): RDD[LabeledPoint] = {

    /** get the file in the spark context */
    val rawData = sc.textFile(hdfs_path)
    
    val data = rawData.map { line =>
      /* Convert the line to an Array of Double */
      val values = line.split(',').map(_.toDouble)
      /** init Returns all elements except the last. */
      /** init Returns all elements except the last. */
      /** Dense creates a dense vector from a double array */
      /** The Vector is a collection type (introduced in Scala 2.8) that addresses the inefficiency for random access on lists. */
      val featureVector = Vectors.dense(values.init)
      /** DecisionTree needs labels starting at 0; subtract 1 */
      val label = values.last - 1
      /**
       * LabeledPoint : Class that represents the features and labels of a data point.
       * label : Label for this data point.
       * features : List of features for this data point.
       */
      LabeledPoint(label, featureVector)
    }
    return data

  }

}