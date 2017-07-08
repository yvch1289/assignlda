
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._

import org.apache.spark.mllib.clustering.{DistributedLDAModel, EMLDAOptimizer, LDA, OnlineLDAOptimizer}

import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.log4j.{Level, Logger}


/**
  * Created by dodge on 07/07/2017.
  */
object Recommendation extends Recommendation {

  private case class Params(
                             input: Seq[String] = Seq.empty,
                             k: Int = 20,
                             maxIterations: Int = 10,
                             docConcentration: Double = -1,
                             topicConcentration: Double = -1,
                             vocabSize: Int = 1000,
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
      .format("com.databricks.spark.xml")
      .option("rowTag", "article")
      .load("C:/Users/dodge/Downloads/dblp.xml/input.xml")

    val selectedData = df.select("author", "title")


    // Load documents, and prepare them for LDA.
    val preprocessStart = System.nanoTime()

    //merge the author into the title docs in order to maintain relationship, adding a suffix to name to retrieve them easily from word to topic distrib/weight
    val my_udf = udf { (arra: Seq[Row], title: String) => if (arra != null && title != null) arra.map(r => r.get(0).asInstanceOf[String].filterNot((x: Char) => x == '.' || x.isWhitespace).concat("Authz")).mkString(" ") + " " + title.filterNot((x: Char) => x == '.') else "" }


    // val my_udf = udf { (arra: Seq[Row], title:String) => if (arra!=null && title!=null ) arra.map(r=> r.get(0).asInstanceOf[String].filterNot((x: Char) => (x.isWhitespace || x==".")).concat("Authz")).mkString(" ") + " "+title else ""}

    val ldainput = selectedData.withColumn("autz", my_udf($"author", $"title")).select($"autz")

    val tokenizer = new RegexTokenizer()
      .setInputCol("autz")
      .setOutputCol("rawTokens")
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("rawTokens")
      .setOutputCol("tokens")
    stopWordsRemover.setStopWords(stopWordsRemover.getStopWords)
    val countVectorizer = new CountVectorizer()
      .setVocabSize(params.vocabSize)
      .setInputCol("tokens")
      .setOutputCol("features")

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, countVectorizer))

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

    lda.setOptimizer(optimizer)
      .setK(params.k)
      .setMaxIterations(params.maxIterations)
      .setDocConcentration(params.docConcentration)
      .setTopicConcentration(params.topicConcentration)
      .setCheckpointInterval(params.checkpointInterval)

    val startTime = System.nanoTime()
    val ldaModel = lda.run(corpus)
    val elapsed = (System.nanoTime() - startTime) / 1e9

    println(s" Summary:")
    println(s"\t time spent: $elapsed sec")

    if (ldaModel.isInstanceOf[DistributedLDAModel]) {
      val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
      val avgLogLikelihood = distLDAModel.logLikelihood / actualCorpusSize.toDouble
      println(s"\t Training data average log likelihood: $avgLogLikelihood")
      println()
    }

    // Print the topics, showing the top-weighted terms for each topic.
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 40)
    val topics = topicIndices.map { case (terms, termWeights) =>
      terms.zip(termWeights).map { case (term, weight) => (vocabArray(term.toInt), weight) }
    }


    val exPerTopic: Array[(Int, Array[(Double, String)])] = topics.zipWithIndex.map { case (topic, i) =>
      println(s"TOPIC $i")

      val topicExperts: Array[(Double, String)] = topic.map { case (term: String, weight: Double) =>
        if (term.contains("authz")) {
          val exp = term.substring(0, term.indexOf("authz"))
          println(s"!!!expert is :$exp")
          (weight, term)
        }
        else {
          println(s"$term\t$weight")
          (0.toDouble, " ")
        }
      }

      println()
      (i, topicExperts)
    }

    // exPerTopic.foreach{case(index,mapOfExperts) => if(index != null && mapOfExperts!=null && !mapOfExperts.isEmpty) mapOfExperts.foreach{case (weight, term) => println(" Topic "+index + " expert = "+ weight + term)} }

    val scanner = new java.util.Scanner(System.in)
    while (true) {


      print("What is  your preference? ")
      val input = scanner.nextLine()
      topics.zipWithIndex.map { case (topic, i) => topic.map { case (term, weight) => if (term.equalsIgnoreCase(input)) {
        (i, weight)
      }
      }
      }

    }

    // sparkSession.stop()

  }


}


class Recommendation extends Serializable {


}
