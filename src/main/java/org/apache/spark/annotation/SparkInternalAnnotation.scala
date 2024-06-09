package org.apache.spark.annotation

object SparkInternalAnnotation {

}

/**
 * Since is not accessible outside of the org.apache.spark package.
 * Therefore, we use an encapsulation violation pattern.
 */
class CustomSince(version: String) extends Since(version)
