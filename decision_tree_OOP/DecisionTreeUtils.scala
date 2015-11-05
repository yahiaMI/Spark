

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD

class DecisionTreeUtils {
  // contains function uses for the decision tree
  
  
  /** computes standard metrics of a DecisionTreeModel: measure the quality of the predictions from a classifier */
  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {

    val predictionsAndLabels = data.map(

      example => (model.predict(example.features), example.label))

    new MulticlassMetrics(predictionsAndLabels)

  }

  /** Allows to calculate the probabilty for each class in the target */
  def classProbabilities(data: RDD[LabeledPoint]): Array[Double] = {

    /** Count (category,count) in data */
    val countsByCategory = data.map(_.label).countByValue()

    /** Order counts by category and extract counts */
    val counts = countsByCategory.toArray.sortBy(_._1).map(_._2)
    counts.map(_.toDouble / counts.sum)
  }

}