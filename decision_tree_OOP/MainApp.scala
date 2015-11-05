

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.DecisionTree

/**
 *
 * The data set, available online at https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/ as a compressed
 * CSV-format data file, covtype.data.gz, and accompanying info file, covtype.info
 *
 * The data set records the types of forest covering parcels of land in Colorado, USA.
 *
 * The target is the forest cover type designation.
 *
 * This script include these different actions :
 *
 * - Prepare the data
 * - Train a Decision Tree model
 * - Create its metrics
 * - Display precision and recall for each target category
 * - check the Random guessing versus the Decision Tree model
 *
 *
 * It's developped under clouder quickstart VM 5.3.0 with Eclipse Scala IDE
 *
 * Referenced Libraries added are :
 * /usr/lib/spark/lib/spark-assembly-1.2.0-cdh5.3.0-hadoop2.5.0-cdh5.3.0.jar
 * /usr/lib/hadoop/client-0.20
 *
 *
 */


object MainApp extends App {

  /********************** Spark configuration  **************************************************************/  
  val conf = new SparkConf()
  conf.setAppName("Spark Connection testing")
  conf.setMaster("spark://quickstart.cloudera:7077")

  /********************** Preparing the data  **************************************************************/
  val sc = new SparkContext(conf)
  val hdfs_path = "hdfs://quickstart.cloudera:8020/user/covtype.data"
  val ET = new ExtractTransform();
  val data = ET.getData(sc, hdfs_path);

  /** Splitting the data, randomly, into the full three subsets: training, cross-validation (CV) and test.*/
  val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))

  /** Caching the data */
  trainData.cache()
  cvData.cache()
  testData.cache()

  /********************** Building the model  **************************************************************/

  // Train a DecisionTree model.
  // Empty categoricalFeaturesInfo indicates all features are continuous.
  val numClasses = 7
  val categoricalFeaturesInfo = Map[Int, Int]()
  val impurity = "gini"
  val maxDepth = 4
  val maxBins = 100

  /** We use of trainClassifier instead of trainRegressor suggests that the target value within each LabeledPoint should be treated as a distinct category number, not anumeric feature value. */

  val model = DecisionTree.trainClassifier(trainData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

  val DTU = new DecisionTreeUtils()
  val metrics = DTU.getMetrics(model, cvData)

  /** Displaying the confusion matrix */

  println("DecisionTree model Confusion Matrix : ")
  println(metrics.confusionMatrix)

  /** Displaying precision */

  println("DecisionTree model precision : ")
  println(metrics.precision)

  /** precision and recall for each category versus the rest */
  println("computing precision and recall for each category versus the rest")
  (0 until 7).map(
    cat => (metrics.precision(cat), metrics.recall(cat))).foreach(println)

  /** We check the Random guessing versus the model **/
  val trainPriorProbabilities = DTU.classProbabilities(trainData)
  val cvPriorProbabilities = DTU.classProbabilities(cvData)

  /** Simulating a random guessing */
  println("Simulating a random guessing")
  val random_guessing_prob = trainPriorProbabilities.zip(cvPriorProbabilities).map {
    /** Calculate the probabilty foreach category */
    case (trainProb, cvProb) => trainProb * cvProb
    /** Then, we sum this for all the model */
  }.sum

  println("random_guessing_prob = ", random_guessing_prob)
  println("Random guessing achieves 37% accuracy then, which makes 70% seem like a good result after all")

}