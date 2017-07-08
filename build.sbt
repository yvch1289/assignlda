name := "mentors-recom"

version := "1.0"

scalaVersion := "2.11.0"

libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.1.1" % "provided"

libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.1.1"


libraryDependencies += "com.databricks" % "spark-xml_2.11" % "0.4.1"

libraryDependencies += "com.github.scopt" % "scopt_2.11" % "3.6.0"