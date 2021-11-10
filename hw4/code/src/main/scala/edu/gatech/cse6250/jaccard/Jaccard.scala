/**
 *
 * students: please put your implementation in this file!
 */
package edu.gatech.cse6250.jaccard

import edu.gatech.cse6250.model._
import edu.gatech.cse6250.model.{ EdgeProperty, VertexProperty }
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object Jaccard {

  def jaccardSimilarityOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long): List[Long] = {
    /**
     * Given a patient ID, compute the Jaccard similarity w.r.t. to all other patients.
     * Return a List of top 10 patient IDs ordered by the highest to the lowest similarity.
     * For ties, random order is okay. The given patientID should be excluded from the result.
     */

    val bi_direction_graph = graph.subgraph(vpred = (x, y) => y.isInstanceOf[PatientProperty]).collectNeighborIds(EdgeDirection.Out).map(f => f._1).collect().toSet
    val graph_neighbors = graph.collectNeighborIds(EdgeDirection.Out)
    val filter_graph_neighbors = graph_neighbors.filter(f => bi_direction_graph.contains(f._1) && f._1.toLong != patientID)
    val neighbor_set = graph_neighbors.filter(f => f._1.toLong == patientID).map(f => f._2).flatMap(f => f).collect().toSet
    val patient_score = filter_graph_neighbors.map(f => (f._1, jaccard(neighbor_set, f._2.toSet)))
    patient_score.takeOrdered(10)(Ordering[Double].reverse.on(f => f._2)).map(_._1.toLong).toList
  }

  def jaccardSimilarityAllPatients(graph: Graph[VertexProperty, EdgeProperty]): RDD[(Long, Long, Double)] = {
    /**
     * Given a patient, med, diag, lab graph, calculate pairwise similarity between all
     * patients. Return a RDD of (patient-1-id, patient-2-id, similarity) where
     * patient-1-id < patient-2-id to avoid duplications
     */

    val sc = graph.edges.sparkContext
    /** Remove this placeholder and implement your code */
    val bi_direction_graph = graph.subgraph(vpred = (x, y) => y.isInstanceOf[PatientProperty]).collectNeighborIds(EdgeDirection.Out).map(f => f._1).collect().toSet
    val patient_neighbor = graph.collectNeighborIds(EdgeDirection.Out).filter(f => bi_direction_graph.contains(f._1))
    val all_neighbor = patient_neighbor.cartesian(patient_neighbor).filter(f => f._1._1 < f._2._1)
    all_neighbor.map(f => (f._1._1, f._2._1, jaccard(f._1._2.toSet, f._2._2.toSet)))
  }

  def jaccard[A](a: Set[A], b: Set[A]): Double = {
    /**
     * Helper function
     *
     * Given two sets, compute its Jaccard similarity and return its result.
     * If the union part is zero, then return 0.
     */

    /** Remove this placeholder and implement your code */
    if (a.isEmpty || b.isEmpty) { return 0.0 }
    a.intersect(b).size.toDouble / a.union(b).size.toDouble
  }
}
