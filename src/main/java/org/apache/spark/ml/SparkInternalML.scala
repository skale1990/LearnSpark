package org.apache.spark.ml

import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.{Vector, VectorUDT}
import org.apache.spark.ml.param.shared.HasParallelism
import org.apache.spark.ml.tuning.ValidatorParams
import org.apache.spark.ml.util._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructField
import org.apache.spark.util.VersionUtils
import org.apache.spark.util.collection.{OpenHashMap, OpenHashSet}

import scala.concurrent.ExecutionContext
import scala.reflect.ClassTag

/**
  * Below objects are not accessible outside of the org.apache.spark.ml package.
  * Therefore, we use an encapsulation violation pattern.
  */
object SparkInternalML {
  def getDefaultParamsReader: DefaultParamsReader.type = DefaultParamsReader
  def getDefaultParamsWriter: DefaultParamsWriter.type = DefaultParamsWriter
  def getMetaAlgorithmReadWrite: MetaAlgorithmReadWrite.type = MetaAlgorithmReadWrite
  def getInstrumentation: Instrumentation.type = Instrumentation
  def getValidatorParams: ValidatorParams.type = ValidatorParams
  def getSchemaUtils: SchemaUtils.type = SchemaUtils
  def getMetadataUtils: MetadataUtils.type = MetadataUtils
  def getVersionUtils: VersionUtils.type = VersionUtils
}

object CustomMetadataUtils {

  /**
   * Obtain the number of features in a vector column.
   * If no metadata is available, extract it from the dataset.
   */
  def getNumFeatures(dataset: Dataset[_], vectorCol: String): Int = {
    getNumFeatures(dataset.schema(vectorCol)).getOrElse {
      dataset.select(DatasetUtils.columnToVector(dataset, vectorCol))
        .head.getAs[Vector](0).size
    }
  }

  /**
   * Examine a schema to identify the number of features in a vector column.
   * Returns None if the number of features is not specified.
   */
  def getNumFeatures(vectorSchema: StructField): Option[Int] = {
    if (vectorSchema.dataType == new VectorUDT) {
      val group = AttributeGroup.fromStructField(vectorSchema)
      val size = group.size
      if (size >= 0) {
        Some(size)
      } else {
        None
      }
    } else {
      None
    }
  }
}

/**
  * ValidatorParams is not accessible outside of the org.apache.spark.ml package.
  * Therefore, we use an encapsulation violation pattern.
  */
trait CustomValidatorParams extends ValidatorParams

/**
  * HasParallelism is not accessible outside of the org.apache.spark.ml package.
  * Therefore, we use an encapsulation violation pattern.
  */
trait CustomHasParallelism extends HasParallelism {
  override def getExecutionContext: ExecutionContext = {
    super.getExecutionContext
  }
}

/**
  * OpenHashMap is not accessible outside of the org.apache.spark package.
  * Therefore, we use an encapsulation violation pattern.
  */
class CustomOpenHashMap[K : ClassTag, V: ClassTag](initialCapacity: Int)
  extends OpenHashMap[K, V](initialCapacity) {
  def this() = this(64)
}

/**
  * OpenHashSet is not accessible outside of the org.apache.spark package.
  * Therefore, we use an encapsulation violation pattern.
  */
class CustomOpenHashSet[T: ClassTag](initialCapacity: Int, loadFactor: Double)
  extends OpenHashSet[T](initialCapacity, loadFactor) {
  def this(initialCapacity: Int) = this(initialCapacity, 0.7)

  def this() = this(64)
}
