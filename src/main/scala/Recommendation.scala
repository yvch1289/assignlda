import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.clustering._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.log4j.{Level, Logger}

/**
  * Created by dodgee on 07/07/2017.
  */
object Recommendation extends Recommendation {

  private case class Params(
                             input: Seq[String] = Seq.empty,
                             k: Int = 20,
                             maxIterations: Int = 10,
                             docConcentration: Double = -1,
                             topicConcentration: Double = -1,
                             vocabSize: Int = 100,
                             algorithm: String = "em",
                             checkpointDir: Option[String] = None,
                             checkpointInterval: Int = 10)

  /** Main function */
  def main(args: Array[String]): Unit = {

    Logger.getRootLogger.setLevel(Level.WARN)
    System.setProperty("hadoop.home.dir", "C:/Users/dodge/winutils")
    val params = Params()
    val sparkSession = SparkSession.builder
      .master("local")
      .appName("example")
      .getOrCreate()
    import sparkSession.implicits._

    val df = sparkSession.read
      .format("com.databricks.spark.xml").option("rowTag", "article")//.option("mode", "DROPMALFORMED") //.option("excludeAttribute", "true")//.option("mode","DROPMALFORMED")
      .load("C:/Users/dodge/Downloads/dblp.xml/dblp.xml")//"C:/Users/dodge/Downloads/dblp.xml/dblp.xml.gz")


    val selectedData = df.select("author", "title")
    println(selectedData.printSchema())
    // Load documents, and prepare them for LDA.
    val preprocessStart = System.nanoTime()

    //merge the author into the title docs in order to maintain relationship, adding a suffix to name to retrieve them easily from word to topic distrib/weight
    val my_udf = udf{

      (arra: Seq[Row], title: Row) =>

          if (arra != null && title != null)
            arra.map(r => r.get(0).asInstanceOf[String].filterNot((x: Char) => x == '.' || x.isWhitespace)
              .concat("Authz")).mkString(" ") + " " + (title.get(0).asInstanceOf[String] match {
              case null => ""
              case x => x}).filterNot((x: Char) => x == '.') else ""

    }

    val ldainput = selectedData.withColumn("merged", my_udf($"author", $"title")) //.select($"autz")

    val tokenizer = new RegexTokenizer().setInputCol("merged").setOutputCol("rawTokens")
    val stopWordsRemover = new StopWordsRemover().setInputCol("rawTokens").setOutputCol("tokens")
    stopWordsRemover.setStopWords(stopWordsRemover.getStopWords)
    val countVectorizer = new CountVectorizer().setVocabSize(params.vocabSize).setInputCol("tokens").setOutputCol("features")

    val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, countVectorizer))

    val model = pipeline.fit(ldainput)
    val documents = model.transform(ldainput)
      .select("features")
      .rdd
      .map { case Row(features: MLVector) => Vectors.fromML(features) }
      .zipWithIndex()
      .map(_.swap)

    val vocab: Array[String] = model.stages(2).asInstanceOf[CountVectorizerModel].vocabulary

    val tokenCount = documents.map(_._2.numActives).sum().toLong // total token count

    val (corpus, vocabArray, actualNumTokens) = (documents, vocab, tokenCount)
    corpus.cache()

    val actualCorpusSize = corpus.count()
    val actualVocabSize = vocabArray.length
    val preprocessElapsed = (System.nanoTime() - preprocessStart) / 1e9

    val lda = new LDA()

    val optimizer = new EMLDAOptimizer

    lda.setOptimizer(optimizer).setK(params.k).setMaxIterations(params.maxIterations).setDocConcentration(params.docConcentration)
      .setTopicConcentration(params.topicConcentration)
      .setCheckpointInterval(params.checkpointInterval)

    val startTime = System.nanoTime()
    val ldaModel = lda.run(corpus)
    val elapsed = (System.nanoTime() - startTime) / 1e9

    println(s" Summary:")
    println(s"\t time spent: $elapsed sec")

    //if (ldaModel.isInstanceOf[DistributedLDAModel]) {
    val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
    val avgLogLikelihood = distLDAModel.logLikelihood / actualCorpusSize.toDouble
    println(s"\t Training data average log likelihood: $avgLogLikelihood")
    println()
    //  }

    // Print the topics, showing the top-weighted terms for each topic.
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 40)
    val topics = topicIndices.map { case (terms, termWeights) =>
      terms.zip(termWeights).map { case (term, weight) => (vocabArray(term.toInt), weight) }
    }

    //build the Topic to Expert Map
    val exPerTopic: Map[Int, Seq[Option[(Double, String)]]] = topics.zipWithIndex.map { case (topic, i) =>
      println(s"TOPIC $i")

      val topicExperts: Set[Option[(Double, String)]] = topic.map { case (term: String, weight: Double) =>
        if (term.contains("authz")) {
          val exp = term.substring(0, term.indexOf("authz"))
          println(s"!!!expert is :$exp")
          Some(weight, term)
        }
        else {
          println(s"$term\t$weight")
          None //(0.toDouble, " ")
        }
      } toSet

      println()
      (i, topicExperts.toSeq.filter(p => p.isDefined).sortWith(_.get._1 > _.get._1))
    } toMap

    val scanner = new java.util.Scanner(System.in)

    while (true) {
      print("What is  your preference? ")
      val input = scanner.nextLine()
      val inp = sparkSession.createDataset(Seq(input))

      val tokenize = new RegexTokenizer().setInputCol("value").setOutputCol("rawTokens")
      val countVectorize = new CountVectorizer().setVocabSize(params.vocabSize).setInputCol("rawTokens").setOutputCol("features")

      val pipelineInp = new Pipeline().setStages(Array(tokenize, countVectorize))

      val mod = pipelineInp.fit(inp)
      val docs = mod.transform(inp).rdd.map { case Row(a, b, features: MLVector) => Vectors.fromML(features) }.zipWithIndex().map(_.swap)

      val topicDistributions = distLDAModel.toLocal.topicDistributions(docs)
      val topicNbr = topicDistributions.first._2.toArray.zipWithIndex.maxBy(_._1)._2

      val expertArr = exPerTopic.get(topicNbr).getOrElse(null)

      if (expertArr != null) {
        if (expertArr.isEmpty) {
          println("did not find Mentor due to no author making it into most weighted words")
        }
        expertArr.map {
          op =>
            op match {
              case Some(a) => {
                println("your mentor can be :" + a._2.substring(0, a._2.indexOf("authz")) + " weight = " + a._1)
              }
              case _ =>
            }
        }
      }

      else
        println("Sorry, no mentor was associated")
    }
    sparkSession.stop()
  }

}


class Recommendation extends Serializable {


}
