package edu.gatech.cse6250.phenotyping

import edu.gatech.cse6250.model.{ Diagnostic, LabResult, Medication }
import org.apache.spark.rdd.RDD

/**
 * @author Hang Su <hangsu@gatech.edu>,
 * @author Sungtae An <stan84@gatech.edu>,
 */
object T2dmPhenotype {

  /** Hard code the criteria */
  val T1DM_DX = Set("250.01", "250.03", "250.11", "250.13", "250.21", "250.23", "250.31", "250.33", "250.41", "250.43",
    "250.51", "250.53", "250.61", "250.63", "250.71", "250.73", "250.81", "250.83", "250.91", "250.93")

  val T2DM_DX = Set("250.3", "250.32", "250.2", "250.22", "250.9", "250.92", "250.8", "250.82", "250.7", "250.72", "250.6",
    "250.62", "250.5", "250.52", "250.4", "250.42", "250.00", "250.02")

  val T1DM_MED = Set("lantus", "insulin glargine", "insulin aspart", "insulin detemir", "insulin lente", "insulin nph", "insulin reg", "insulin,ultralente")

  val T2DM_MED = Set("chlorpropamide", "diabinese", "diabanase", "diabinase", "glipizide", "glucotrol", "glucotrol xl",
    "glucatrol ", "glyburide", "micronase", "glynase", "diabetamide", "diabeta", "glimepiride", "amaryl",
    "repaglinide", "prandin", "nateglinide", "metformin", "rosiglitazone", "pioglitazone", "acarbose",
    "miglitol", "sitagliptin", "exenatide", "tolazamide", "acetohexamide", "troglitazone", "tolbutamide",
    "avandia", "actos", "actos", "glipizide")

  val DM_RELATED_DX = Set("790.21", "790.22", "790.2", "790.29", "648.81", "648.82", "648.83", "648.84", "648", "648",
    "648.01", "648.02", "648.03", "648.04", "791.5", "277.7", "V77.1", "256.4")

  /**
   * Transform given data set to a RDD of patients and corresponding phenotype
   *
   * @param medication medication RDD
   * @param labResult  lab result RDD
   * @param diagnostic diagnostic code RDD
   * @return tuple in the format of (patient-ID, label). label = 1 if the patient is case, label = 2 if control, 3 otherwise
   */
  def transform(medication: RDD[Medication], labResult: RDD[LabResult], diagnostic: RDD[Diagnostic]): RDD[(String, Int)] = {
    /**
     * Remove the place holder and implement your code here.
     * Hard code the medication, lab, icd code etc. for phenotypes like example code below.
     * When testing your code, we expect your function to have no side effect,
     * i.e. do NOT read from file or write file
     *
     * You don't need to follow the example placeholder code below exactly, but do have the same return type.
     *
     * Hint: Consider case sensitivity when doing string comparisons.
     */

    val sc = medication.sparkContext
    val patients = diagnostic.map(_.patientID).union(labResult.map(_.patientID)).union(medication.map(_.patientID)).distinct()

    /*Case*/
    val type1DG = diagnostic.filter(f => T1DM_DX.contains(f.code)).map(_.patientID).distinct()
    val type1DGExc = patients.subtract(type1DG).distinct()
    val type2DG = diagnostic.filter(f => T2DM_DX.contains(f.code)).map(_.patientID).distinct()

    val type1MG = medication.filter(f => T1DM_MED.contains(f.medicine.toLowerCase)).map(_.patientID).distinct()
    val type1MGExc = patients.subtract(type1MG).distinct()
    val type2MG = medication.filter(f => T2DM_MED.contains(f.medicine.toLowerCase)).map(_.patientID).distinct()
    val type2MGExc = patients.subtract(type2MG).distinct()

    val p_1 = type1DGExc.intersection(type2DG).intersection(type1MGExc)
    val p_2 = type1DGExc.intersection(type2DG).intersection(type1MG).intersection(type2MGExc)
    val prep_3 = type1DGExc.intersection(type2DG).intersection(type1MG).intersection(type2MG)
    val condition5 = medication.map(f => (f.patientID, f)).join(prep_3.map(f => (f, 0))).map(f => Medication(f._2._1.patientID, f._2._1.date, f._2._1.medicine))
    val condition5_1 = condition5.filter(f => T1DM_MED.contains(f.medicine.toLowerCase)).map(f => (f.patientID, f.date.getTime())).reduceByKey(Math.min)
    val condition5_2 = condition5.filter(f => T2DM_MED.contains(f.medicine.toLowerCase)).map(f => (f.patientID, f.date.getTime())).reduceByKey(Math.min)
    val p_3 = condition5_1.join(condition5_2).filter(f => f._2._1 > f._2._2).map(_._1)
    val casePatients_id = sc.union(p_1, p_2, p_3).distinct()

    //////Control//////
    val glu_contain = labResult.filter(f => f.testName.toLowerCase.contains("glucose"))
    val glu_id = glu_contain.map(_.patientID).distinct()
    val glu_id_list = glu_id.collect.toSet
    val condition1 = labResult.filter(f => glu_id_list(f.patientID))

    val abn1 = condition1.filter(f => f.testName.equals("hba1c") && f.value >= 6.0).map(f => f.patientID)
    val abn2 = condition1.filter(f => f.testName.equals("hemoglobin a1c") && f.value >= 6.0).map(f => f.patientID)
    val abn3 = condition1.filter(f => f.testName.equals("fasting glucose") && f.value >= 110).map(f => f.patientID)
    val abn4 = condition1.filter(f => f.testName.equals("fasting blood glucose") && f.value >= 110).map(f => f.patientID)
    val abn5 = condition1.filter(f => f.testName.equals("fasting plasma glucose") && f.value >= 110).map(f => f.patientID)
    val abn6 = condition1.filter(f => f.testName.equals("glucose") && f.value > 110).map(f => f.patientID)
    val abn7 = condition1.filter(f => f.testName.equals("glucose, serum") && f.value > 110).map(f => f.patientID)
    val AbnormallabValues = sc.union(abn1, abn2, abn3, abn4, abn5, abn6, abn7).distinct()

    val condition2 = glu_id.subtract(AbnormallabValues).distinct()
    val diabetes_1 = diagnostic.filter(f => DM_RELATED_DX.contains(f.code)).map(f => f.patientID).distinct()
    val diabetes_2 = diagnostic.filter(f => f.code.startsWith("250.")).map(f => f.patientID).distinct()
    val condition3 = patients.subtract(diabetes_1.union(diabetes_2)).distinct()
    val controlPatients_id = condition2.intersection(condition3).distinct()

    //////Other//////
    val otherPatients_id = patients.subtract(casePatients_id).subtract(controlPatients_id).distinct()

    /** Hard code the criteria */
    // val type1_dm_dx = Set("code1", "250.03")
    // val type1_dm_med = Set("med1", "insulin nph")
    // use the given criteria above like T1DM_DX, T2DM_DX, T1DM_MED, T2DM_MED and hard code DM_RELATED_DX criteria as well

    /** Find CASE Patients */
    val casePatients = casePatients_id.map((_, 1))

    /** Find CONTROL Patients */
    val controlPatients = controlPatients_id.map((_, 2))

    /** Find OTHER Patients */
    val others = otherPatients_id.map((_, 3))

    /** Once you find patients for each group, make them as a single RDD[(String, Int)] */
    val phenotypeLabel = sc.union(casePatients, controlPatients, others)

    /** Return */
    phenotypeLabel
  }
}