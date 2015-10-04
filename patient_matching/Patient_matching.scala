
/**
 *
 * We’re going to use a sample data set from the UC Irvine Machine Learning Repository.
 * The data set we’ll be analyzing was curated from a record linkage study that was performed at a German hospital in 2010, and it contains several million pairs of patient records that were matched according to several different criteria, such as the patient’s name (first and last), their address, and their birthday. Each matching field was assigned a numerical score from 0.0 to 1.0 based on how similar the strings were, and the data was then hand labeled to identify which pairs represented the same person and which did not.
 *
 * Data sample :
 *
 * "id_1","id_2","cmp_fname_c1","cmp_fname_c2","cmp_lname_c1","cmp_lname_c2","cmp_sex","cmp_bd","cmp_bm","cmp_by","cmp_plz","is_match"
 * 37291,53113,0.833333333333333,?,1,?,1,1,1,1,0,TRUE
 * 39086,47614,1,?,1,?,1,1,1,1,1,TRUE
 * 70031,70237,1,?,1,?,1,1,1,1,1,TRUE
 * 84795,97439,1,?,1,?,1,1,1,1,1,TRUE
 * 36950,42116,1,?,1,1,1,1,1,1,1,TRUE
 * 42413,48491,1,?,1,?,1,1,1,1,1,TRUE
 * 25965,64753,1,?,1,?,1,1,1,1,1,TRUE
 *
 * Source : https://archive.ics.uci.edu/ml/machine-learning-databases/00210/donation.zip
 *
 * Target :
 * - Analyze the differences in the distribution of the arrays of scores for both the matches and the non-matches
 * - Make a scoring function to predict if two patients match or not.
 *
 * Dependencies : 
 * - NAStatCounter.scala
 * - StatsWithMissing.scala
 * 
 */

/************************************************* Bib ***************************************************/



//check the header line
def isHeader(line: String): Boolean = {
  line.contains("id_1")
}

// Check the if ? exists then return a Nan or Double
def toDouble(s: String) = {
  if ("?".equals(s)) Double.NaN else s.toDouble
}

/** Converts Nan to double */
def naz(d: Double) = if (Double.NaN.equals(d)) 0.0 else d

//The class MatchData : Setting a tuple column name
case class MatchData(id1: Int, id2: Int, scores: Array[Double], matched: Boolean)

// to calculate the score
case class Scored(md: MatchData, score: Double)

/** The parse function */
def parse(line: String) = {
  val pieces = line.split(',')
  val id1 = pieces(0).toInt
  val id2 = pieces(1).toInt
  val scores = pieces.slice(2, 11).map(toDouble)
  val matched = pieces(11).toBoolean
  MatchData(id1, id2, scores, matched)
}

/************************************************* Preparing the Data ***************************************************/

/**
 * download the data and add it to the HDFS
 * mkdir linkage
 * cd linkage/
 * curl -o donation.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00210/donation.zip
 * unzip donation.zip
 * unzip 'block*.zip'
 * hadoop fs -mkdir linkage
 * hadoop fs -put block*csv linkage
 *
 */

// getting the csv files, we can get it from the HDFS
val rawblocks = sc.textFile("/mnt/partage/linkage")

// removing the header
val noheader = rawblocks.filter(x => !isHeader(x))

// noheader to tuple
val parsed = noheader.map(line => parse(line))

parsed.cache()

/*************************************************  Computing Summary Statistics ***************************************************/

/**
 * We use NAStatCounter class to process the scores in the MatchData records
 * within the parsed RDD. Each MatchData instance contains an array of scores of type
 * Array[Double]. For each entry in the array, we would like to have an NAStatCounter
 * instance that tracked how many of the values in that index were NaN along with the
 * regular distribution statistics for the non-missing values.
 */

val nasRDD = parsed.map(
  md =>
    {
      md.scores.map(d => NAStatCounter(d))
    })

//Get the differences in the distribution of the arrays of scores for both the matches and the non-matches

// reduce is an aggregation function 
// We aggregate every column NAStatCounter
val reduced = nasRDD.reduce((n1, n2) => {
  n1.zip(n2).map { case (a, b) => a.merge(b) }
})

reduced.foreach(println)

/*************************************************  Simple Variable Selection and Scoring ***************************************************/

// Get the stats for matched and non matched lines for each column
val statsm = statsWithMissing(parsed.filter(_.matched).map(_.scores))
val statsn = statsWithMissing(parsed.filter(!_.matched).map(_.scores))

//Calculate the means difference  and the missing numberfor each column 

statsm.zip(statsn).map {
  case (m, n) =>
    (m.missing + n.missing, m.stats.mean - n.stats.mean)
}.foreach(println)

/**
 * A good feature has two properties: it tends to have significantly different values for
 * matches and non-matches (so the difference between the means will be large) and it
 * occurs often enough in the data that we can rely on it to be regularly available for any
 * pair of records.
 */

/**
 * Features 5 and 7, on the other hand, are excellent : they almost always occur for any pair
 * of records, and there is a very large difference in the mean values (over 0.77 for both
 * features.) Features 2, 6, and 8 also seem beneficial: they are generally available in the
 * data set and the difference in mean values for matches and non-matches are substantial.
 */

/** Create  RDD of score */

val ct = parsed.map(md => {
  val score = Array(2, 5, 6, 7, 8).map(i => naz(md.scores(i))).sum
  Scored(md, score)
})

/** We use a Threshold. We calculate the number of false and true value according to threshold*/
ct.filter(s => s.score >= 4.0).map(s => s.md.matched).countByValue()











