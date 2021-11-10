package edu.gatech.cse6250.features

import edu.gatech.cse6250.model.{ Diagnostic, LabResult, Medication }
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD

/**
 * @author Hang Su
 */
object FeatureConstruction {

  /**
   * ((patient-id, feature-name), feature-value)
   */
  type FeatureTuple = ((String, String), Double)

  /**
   * Aggregate feature tuples from diagnostic with COUNT aggregation,
   *
   * @param diagnostic RDD of diagnostic
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val diagnostic_df = diagnostic.map(f => ((f.patientID, f.code), 1.0)).reduceByKey(_ + _)

    diagnostic_df

  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation,
   *
   * @param medication RDD of medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val medication_df = medication.map(f => ((f.patientID, f.medicine), 1.0)).reduceByKey(_ + _)

    medication_df
  }

  /**
   * Aggregate feature tuples from lab result, using AVERAGE aggregation
   *
   * @param labResult RDD of lab result
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val labResult_1 = labResult.map(f => ((f.patientID, f.testName), 1.0)).reduceByKey(_ + _)
    val labResult_2 = labResult.map(f => ((f.patientID, f.testName), f.value)).reduceByKey(_ + _)
    val labResult_df = labResult_2.join(labResult_1).map(f => (f._1, f._2._1 / f._2._2))

    labResult_df
  }

  /**
   * Aggregate feature tuple from diagnostics with COUNT aggregation, but use code that is
   * available in the given set only and drop all others.
   *
   * @param diagnostic   RDD of diagnostics
   * @param candiateCode set of candidate code, filter diagnostics based on this set
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic], candiateCode: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val diagnostic_df = diagnostic.map(f => ((f.patientID, f.code), 1.0)).reduceByKey(_ + _)
    val diagnostic_p_df = diagnostic_df.filter(f => (candiateCode.contains(f._1._2)))

    diagnostic_p_df
  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation, use medications from
   * given set only and drop all others.
   *
   * @param medication          RDD of diagnostics
   * @param candidateMedication set of candidate medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication], candidateMedication: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val medication_df = medication.map(f => ((f.patientID, f.medicine), 1.0)).reduceByKey(_ + _)
    val medication_p_df = medication_df.filter(f => (candidateMedication.contains(f._1._2)))

    medication_p_df
  }

  /**
   * Aggregate feature tuples from lab result with AVERAGE aggregation, use lab from
   * given set of lab test names only and drop all others.
   *
   * @param labResult    RDD of lab result
   * @param candidateLab set of candidate lab test name
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult], candidateLab: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val labResult_1 = labResult.map(f => ((f.patientID, f.testName), 1.0)).reduceByKey(_ + _)
    val labResult_2 = labResult.map(f => ((f.patientID, f.testName), f.value)).reduceByKey(_ + _)
    val labResult_df = labResult_2.join(labResult_1).map(f => (f._1, f._2._1 / f._2._2))
    val labResult_p_df = labResult_df.filter(f => (candidateLab.contains(f._1._2)))

    labResult_p_df
  }

  /**
   * Given a feature tuples RDD, construct features in vector
   * format for each patient. feature name should be mapped
   * to some index and convert to sparse feature format.
   *
   * @param sc      SparkContext to run
   * @param feature RDD of input feature tuples
   * @return
   */
  def construct(sc: SparkContext, feature: RDD[FeatureTuple]): RDD[(String, Vector)] = {

    /** save for later usage */
    feature.cache()

    /** create a feature name to id map */
    val index_feature = feature.map(_._1._2).distinct().collect.zipWithIndex.toMap
    val feature_map = sc.broadcast(index_feature)
    /** transform input feature */
    val IdWithFeature = feature.map(f => (f._1._1, (f._1._2, f._2))).groupByKey()
    val result = IdWithFeature.map {
      case (target, features) =>
        val nums = feature_map.value.size
        val index_num = features.toList.map {
          case (name, values) =>
            (feature_map.value(name), values)
        }
        val vector_num = Vectors.sparse(nums, index_num)
        val labels = (target, vector_num)

        labels
    }
    /**
     * Functions maybe helpful:
     * collect
     * groupByKey
     */

    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    result

    /** The feature vectors returned can be sparse or dense. It is advisable to use sparse */

  }
}

