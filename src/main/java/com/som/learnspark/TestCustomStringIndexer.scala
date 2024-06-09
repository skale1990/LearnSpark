package com.som.learnspark

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
//import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, Row, SparkSession}
import org.apache.spark.sql.types.{IntegerType, StructType}

object TestCustomStringIndexer {

  private val spark: SparkSession = SparkSession.builder().master("local[2]")
    .appName("TestSuite")
    .config("spark.sql.shuffle.partitions", "2")
//    .config("spark.sql.parser.quotedRegexColumnNames", value = true)
    .getOrCreate()

  def main(args: Array[String]): Unit = {
//    import spark.implicits._
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val structureData = Seq(
      Row(Row(10, 12), 1000),
      Row(Row(12, 14), 4300),
      Row( Row(37, 891), 1400),
      Row(Row(8902, 12), 4000),
      Row(Row(12, 89), 1000)
    )

    val structureSchema = new StructType()
      .add("location", new StructType()
        .add("longitude", IntegerType)
        .add("latitude", IntegerType))
      .add("salary", IntegerType)
    val df = spark.createDataFrame(spark.sparkContext.parallelize(structureData), structureSchema)

    def flattenSchema(schema: StructType, prefix: String = null, prefixSelect: String = null):
    Array[Column] = {
      schema.fields.flatMap(f => {
        val colName = if (prefix == null) f.name else (prefix + "." + f.name)
        val colnameSelect = if (prefix == null) f.name else (prefixSelect + "." + f.name)

        f.dataType match {
          case st: StructType => flattenSchema(st, colName, colnameSelect)
          case _ =>
            Array(col(colName).as(colnameSelect))
        }
      })
    }

    val flattenColumns = flattenSchema(df.schema)
    val flattenedDf = df.select(flattenColumns: _*)

    flattenedDf.printSchema
    flattenedDf.show()

//    val renameColumn = new RenameColumn().setInputCol("location.longitude").setOutputCol("location_longitude")
//    val si = new CustomStringIndexer().setInputCol("location_longitude").setOutputCol("longitutdee")
//    val pipeline = new Pipeline().setStages(Array(renameColumn, si))

    val si = new CustomStringIndexer().setInputCol("`location.longitude`").setOutputCol("longitutdee")
    val pipeline = new Pipeline().setStages(Array(si))
    val pipelineRes = pipeline.fit(flattenedDf).transform(flattenedDf)
    pipelineRes.show()

    pipelineRes.explain(true)

    /**
     * +------------------+-----------------+------+-----------+
     * |location_longitude|location.latitude|salary|longitutdee|
     * +------------------+-----------------+------+-----------+
     * |                10|               12|  1000|        1.0|
     * |                12|               14|  4300|        0.0|
     * |                37|              891|  1400|        2.0|
     * |              8902|               12|  4000|        3.0|
     * |                12|               89|  1000|        0.0|
     * +------------------+-----------------+------+-----------+
     */
  }
}
