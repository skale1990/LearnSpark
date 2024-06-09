package com.som.learnspark


import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{StructField, StructType}

class RenameColumn(val uid: String) extends Transformer with Params
  with HasInputCol with HasOutputCol with DefaultParamsWritable  {
  def this() = this(Identifiable.randomUID("RenameColumn"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  def validateAndTransformSchema(schema: StructType): StructType = {
    val col = schema(getInputCol)
    schema.add(StructField(getOutputCol, col.dataType, col.nullable, col.metadata))
  }

  def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  def copy(extra: ParamMap): RenameColumn = defaultCopy(extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    dataset.toDF().withColumnRenamed(getInputCol, getOutputCol)
  }
}

object RenameColumn extends DefaultParamsReadable[RenameColumn] {
  override def load(path: String): RenameColumn = super.load(path)
}

