package edu.gatech.cse6250.randomwalk

import edu.gatech.cse6250.model.{ PatientProperty, EdgeProperty, VertexProperty }
import org.apache.spark.graphx._

object RandomWalk {

  def randomWalkOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long, numIter: Int = 100, alpha: Double = 0.15): List[Long] = {
    /**
     * Given a patient ID, compute the random walk probability w.r.t. to all other patients.
     * Return a List of patient IDs ordered by the highest to the lowest similarity.
     * For ties, random order is okay
     */

    /** Initialize the PageRank graph with each edge **/
    val initial_p = patientID
    val vertex_p: VertexId = patientID

    /** When running page_rank_random, all others are set to 0 and the source vertex has an attribute 1.0 **/
    var graph_rank_random: Graph[Double, Double] = graph.outerJoinVertices(graph.outDegrees) { (x, y, z) => z.getOrElse(0) }.mapTriplets(f => 1.0 / f.srcAttr, TripletFields.Src).mapVertices { (x, y) => if (!(x != vertex_p)) 1.0 else 0.0 }

    def diff(n1: VertexId, n2: VertexId): Double = { if (n1 == n2) 1.0 else 0.0 }

    var inter = 0
    var grapg_rankv1: Graph[Double, Double] = null

    while (inter < numIter) {
      graph_rank_random.cache()
      val updatev1 = graph_rank_random.aggregateMessages[Double](f => f.sendToDst(f.srcAttr * f.attr), _ + _, TripletFields.Src)

      grapg_rankv1 = graph_rank_random

      val updatev2 = { (vertex_p: VertexId, vertex_n: VertexId) => alpha * diff(vertex_p, vertex_n) }

      graph_rank_random = graph_rank_random.outerJoinVertices(updatev1) { (x, y, z) => updatev2(vertex_p, x) + (1.0 - alpha) * z.getOrElse(0.0) }.cache()
      graph_rank_random.edges.foreachPartition(f => {})
      grapg_rankv1.vertices.unpersist(false)
      grapg_rankv1.edges.unpersist(false)

      inter = inter + 1
    }

    val RWgraph = graph.subgraph(vpred = (x, y) => y.isInstanceOf[PatientProperty]).collectNeighborIds(EdgeDirection.Out).map(f => f._1).collect().toSet
    val RWgraph_top10 = graph_rank_random.vertices.filter(f => RWgraph.contains(f._1)).takeOrdered(11)(Ordering[Double].reverse.on(f => f._2)).map(_._1)
    RWgraph_top10.slice(1, RWgraph_top10.length).toList
  }
}
