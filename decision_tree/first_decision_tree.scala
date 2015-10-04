import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.evaluation._
/**
 *
 * The data set, available online at https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/ as a compressed
 * CSV-format data file, covtype.data.gz, and accompanying info file, covtype.info
 *
 * The data set records the types of forest covering parcels of land in Colorado, USA.
 *
 * The target is the forest cover type designation.
 *
 * This script include these differents actions :
 *
 * - Prepare the data
 * - Train a Decision Tree model
 * - Create its metrics
 * - Display precision and recall for each target category
 * - check the Random guessing versus the Decision Tree model
 * - Decision Tree Hyperparameters random guessing
 * - Train a DecisionTree model with other parameters
 * - Train a Random decision Forest
 * - Computes standard metrics
 * - Predict a vector
 * - computes standard metrics for the Test Set
 *
 *
 */

import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.tree.RandomForest



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

/********************** Preparing the data **************************************************************/

/** get the file in the spark context */
val rawData = sc.textFile("/mnt/partage/covtype.data")

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

/** Splitting the data, randomly, into the full three subsets: training, cross-validation (CV) and test.*/
val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))

/** Caching the data */
trainData.cache()
cvData.cache()
testData.cache()

/********************** Building models **************************************************************/

// Train a DecisionTree model.

//  Empty categoricalFeaturesInfo indicates all features are continuous.
val numClasses = 7
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 4
val maxBins = 100

/** We use of trainClassifier instead of trainRegressor suggests that the target value within each LabeledPoint should be treated as a distinct category number, not anumeric feature value. */
val model = DecisionTree.trainClassifier(trainData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

val metrics = getMetrics(model, cvData)

/** Displaying the confusion matrix */
println("DecisionTree model Confusion Matrix  : ")
println(metrics.confusionMatrix)

/** Displaying precision */
println("DecisionTree model precision  : ")
println(metrics.precision)

/** precision and recall for each category versus the rest */
println("computing precision and recall for each category versus the rest")
(0 until 7).map(
  cat => (metrics.precision(cat), metrics.recall(cat))).foreach(println)

/** We check the Random guessing versus the model **/

val trainPriorProbabilities = classProbabilities(trainData)
val cvPriorProbabilities = classProbabilities(cvData)

/** Simulating a random guessing */
println("Simulating a random guessing")
val random_guessing_prob = trainPriorProbabilities.zip(cvPriorProbabilities).map {

  /** Calculate the probabilty foreach category */
  case (trainProb, cvProb) => trainProb * cvProb
  /** Then, we sum this for all the model */
}.sum

println("random_guessing_prob = ", random_guessing_prob)
println("Random guessing achieves 37% accuracy then, which makes 70% seem like a good result after all")

/** The Decision Tree Hyperparameters random guessing below requires a lot of memory */
/**
 * val evaluations =
 * for (
 * impurity <- Array("gini", "entropy");
 * depth <- Array(1, 20);
 * bins <- Array(10, 300)
 * )
 * yield {
 * val model = DecisionTree.trainClassifier(trainData, 7, Map[Int,Int](), impurity, depth, bins)
 * val predictionsAndLabels = cvData.map(
 * example => (model.predict(example.features), example.label)
 * )
 * val accuracy = new MulticlassMetrics(predictionsAndLabels).precision
 * ((impurity, depth, bins), accuracy)
 * }
 *
 * evaluations.sortBy(_._2).reverse.foreach(println)
 *
 */

// Train a DecisionTree model with other parameters 

val numClasses = 7
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "entropy"
val maxDepth = 20
val maxBins = 30

val model = DecisionTree.trainClassifier(trainData.union(cvData), numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
val metrics = getMetrics(model, trainData.union(cvData))

println("metrics.precision = ", metrics.precision)

/** Train a Random decision Forest*/

/**
 * Random decision forests are appealing in the context of big data because trees are sup‐
 * posed to be built independently, and, big-data technologies like Spark and MapReduce
 * inherently need data-parallel problems
 */

/* trees are built on just a subset of all training data and can be internally cross-validated against the remaining data */
val numClasses = 7
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 3 // 20 to use more in practice.
val featureSubsetStrategy = "auto" // We let the algorithm choose.
val impurity = "entropy"
val maxDepth = 30
val maxBins = 300

val forest = RandomForest.trainClassifier(trainData, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

/** computes standard metrics : measure the quality of the predictions from a classifier */
val predictionsAndLabels = cvData.map(example =>
  (forest.predict(example.features), example.label))

val metrics = new MulticlassMetrics(predictionsAndLabels)
println("Random forest model : metrics Precision for CV set= ", metrics.precision)

/********************** Predictions **************************************************************/

/** How to predict features? */

val vector = testData.first.features

println("vector", testData.first.features)
println(" Random decision forests model prediction of the vector", forest.predict(vector))
println(" Decision tree model prediction of the vector", model.predict(vector))

// display precision foreach model

val predictionsAndLabels = testData.map(example =>
  (forest.predict(example.features), example.label))

val metricsDF = new MulticlassMetrics(predictionsAndLabels)
println("Random forest model : metrics Precision for the test set = ", metrics.precision)

val metricsDT = getMetrics(model, testData)
println("Decision Tree model : metrics Precision for the test set = ", metrics.precision)



 










