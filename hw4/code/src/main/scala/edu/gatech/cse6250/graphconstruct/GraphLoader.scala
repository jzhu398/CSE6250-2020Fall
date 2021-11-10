/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse6250.graphconstruct

import edu.gatech.cse6250.model._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object GraphLoader {
  /**
   * Generate Bipartite Graph using RDDs
   *
   * @input: RDDs for Patient, LabResult, Medication, and Diagnostic
   * @return: Constructed Graph
   *
   */
  def load(patients: RDD[PatientProperty], labResults: RDD[LabResult],
    medications: RDD[Medication], diagnostics: RDD[Diagnostic]): Graph[VertexProperty, EdgeProperty] = {

    val sc = patients.sparkContext
    val patient_vertex: RDD[(VertexId, VertexProperty)] = patients.map(patient => (patient.patientID.toLong, patient.asInstanceOf[VertexProperty]))
    val num_patient = patients.map(f => f.patientID).distinct().count()

    /** diag vertex **/
    val latest_diag_event = diagnostics.map(f => ((f.patientID, f.icd9code), f)).reduceByKey((f1, f2) => if (f1.date > f2.date) f1 else f2).map { case (key, x) => x }
    val diag_node = latest_diag_event.map(_.icd9code).distinct().zipWithIndex().map { case (uniq, uniq2) => (uniq, uniq2 + num_patient + 1) }
    val Diag_Vert_Id = diag_node.collect.toMap
    val diag_vertex: RDD[(VertexId, VertexProperty)] = diag_node.map { case (icd9code, values) => (values, DiagnosticProperty(icd9code)) }
    val num_diag = diag_node.count()

    /** lab vertex **/
    val latest_lab_event = labResults.map(f => ((f.patientID, f.labName), f)).reduceByKey((f1, f2) => if (f1.date > f2.date) f1 else f2).map { case (key, x) => x }
    val lab_node = latest_lab_event.map(_.labName).distinct().zipWithIndex().map { case (uniq, uniq2) => (uniq, uniq2 + num_patient + 1 + num_diag) }
    val Lab_Vert_Id = lab_node.collect.toMap
    val lab_vertex: RDD[(VertexId, VertexProperty)] = lab_node.map { case (labName, values) => (values, LabResultProperty(labName)) }
    val num_lab = lab_node.count()

    /** med vertex **/
    val latest_med_event = medications.map(f => ((f.patientID, f.medicine), f)).reduceByKey((f1, f2) => if (f1.date > f2.date) f1 else f2).map { case (key, x) => x }
    val med_node = latest_med_event.map(_.medicine).distinct().zipWithIndex().map { case (uniq, uniq2) => (uniq, uniq2 + num_patient + 1 + num_diag + num_lab) }
    val Med_Vert_Id = med_node.collect.toMap
    val med_vertex: RDD[(VertexId, VertexProperty)] = med_node.map { case (medicine, values) => (values, MedicationProperty(medicine)) }
    val num_med = med_node.count()

    /** Define Graph Edges for diag **/
    val gDiag_Vert_Id = sc.broadcast(Diag_Vert_Id)
    val PDEdges = latest_diag_event.map(f => (f.patientID, f.icd9code, f)).map { case (patientID, icd9code, index) => Edge(patientID.toLong, gDiag_Vert_Id.value(icd9code), PatientDiagnosticEdgeProperty(index).asInstanceOf[EdgeProperty]) }
    val DPEdges = latest_diag_event.map(f => (f.patientID, f.icd9code, f)).map { case (patientID, icd9code, index) => Edge(gDiag_Vert_Id.value(icd9code), patientID.toLong, PatientDiagnosticEdgeProperty(index).asInstanceOf[EdgeProperty]) }
    val finalPDEdges = sc.union(PDEdges, DPEdges)

    /** Define Graph Edges for lab **/
    val gLab_Vert_Id = sc.broadcast(Lab_Vert_Id)
    val PLEdges = latest_lab_event.map(f => (f.patientID, f.labName, f)).map { case (patientID, labName, index) => Edge(patientID.toLong, gLab_Vert_Id.value(labName), PatientLabEdgeProperty(index).asInstanceOf[EdgeProperty]) }
    val LPEdges = latest_lab_event.map(f => (f.patientID, f.labName, f)).map { case (patientID, labName, index) => Edge(gLab_Vert_Id.value(labName), patientID.toLong, PatientLabEdgeProperty(index).asInstanceOf[EdgeProperty]) }
    val finalPLEdges = sc.union(PLEdges, LPEdges)

    /** Define Graph Edges for med **/
    val gMed_Vert_Id = sc.broadcast(Med_Vert_Id)
    val PMEdges = latest_med_event.map(f => (f.patientID, f.medicine, f)).map { case (patientID, medicine, index) => Edge(patientID.toLong, gMed_Vert_Id.value(medicine), PatientMedicationEdgeProperty(index).asInstanceOf[EdgeProperty]) }
    val MPEdges = latest_med_event.map(f => (f.patientID, f.medicine, f)).map { case (patientID, medicine, index) => Edge(gMed_Vert_Id.value(medicine), patientID.toLong, PatientMedicationEdgeProperty(index).asInstanceOf[EdgeProperty]) }
    val finalPMEdges = sc.union(PMEdges, MPEdges)

    // Making Graph
    val finalVertices = sc.union(patient_vertex, diag_vertex, lab_vertex, med_vertex)
    val finalEdges = sc.union(finalPDEdges, finalPLEdges, finalPMEdges)
    val graph: Graph[VertexProperty, EdgeProperty] = Graph[VertexProperty, EdgeProperty](finalVertices, finalEdges)

    graph
  }
}
