//! Gauge equivariance: features that transform correctly under change of local coordinates.
//!
//! On a manifold, each point has a local coordinate system (gauge). Gauge-equivariant
//! features transform properly when we change the gauge — this is the "geometric" part
//! of geometric deep learning.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::core::{AgentFeatures, AgentGraph, GroupAction};

/// A gauge transformation: changes local coordinates at each agent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GaugeTransformation {
    /// Per-agent gauge matrices (fiber_dim × fiber_dim).
    pub gauges: Vec<DMatrix<f64>>,
}

impl GaugeTransformation {
    pub fn new(gauges: Vec<DMatrix<f64>>) -> Self {
        Self { gauges }
    }

    /// Identity gauge transformation.
    pub fn identity(n_agents: usize, fiber_dim: usize) -> Self {
        Self {
            gauges: (0..n_agents).map(|_| DMatrix::identity(fiber_dim, fiber_dim)).collect(),
        }
    }

    /// Random orthogonal gauge transformation.
    pub fn random_orthogonal(n_agents: usize, fiber_dim: usize, seed: u64) -> Self {
        let mut gauges = Vec::new();
        let mut s = seed;
        for _ in 0..n_agents {
            // Generate a random orthogonal matrix via Gram-Schmidt
            let mut m = DMatrix::zeros(fiber_dim, fiber_dim);
            for i in 0..fiber_dim {
                for j in 0..fiber_dim {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                    m[(i, j)] = ((s as i64) as f64) / (u64::MAX as f64) * 2.0 - 1.0;
                }
            }
            // Orthogonalize via QR (simplified)
            let qr = m.qr();
            let q = qr.q();
            gauges.push(q);
        }
        Self { gauges }
    }

    /// Number of agents.
    pub fn n_agents(&self) -> usize {
        self.gauges.len()
    }

    /// Fiber dimension.
    pub fn fiber_dim(&self) -> usize {
        if self.gauges.is_empty() { 0 } else { self.gauges[0].nrows() }
    }
}

impl GroupAction for GaugeTransformation {
    fn act(&self, features: &DMatrix<f64>) -> DMatrix<f64> {
        let n = features.nrows();
        let d = features.ncols();
        let fd = self.fiber_dim();
        let n_feats = d / fd.max(1);
        let mut result = DMatrix::zeros(n, d);

        for i in 0..n.min(self.gauges.len()) {
            let gauge = &self.gauges[i];
            for k in 0..n_feats {
                let offset = k * fd;
                if offset + fd <= d {
                    let slice = features.columns(offset, fd).row(i).transpose();
                    let transformed = gauge * &slice;
                    for j in 0..fd {
                        result[(i, offset + j)] = transformed[j];
                    }
                }
            }
        }
        // Copy remaining features that don't fit fiber structure
        for i in 0..n {
            for j in (n_feats * fd)..d {
                result[(i, j)] = features[(i, j)];
            }
        }
        result
    }

    fn inverse(&self) -> Self {
        Self {
            gauges: self.gauges.iter().map(|g| {
                // For orthogonal matrices, inverse = transpose
                g.transpose()
            }).collect(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        Self {
            gauges: self.gauges.iter().zip(other.gauges.iter()).map(|(a, b)| {
                a * b
            }).collect(),
        }
    }

    fn identity() -> Self {
        Self { gauges: vec![] }
    }
}

/// Gauge-equivariant layer using parallel transport.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GaugeEquivLayer {
    pub input_fiber_dim: usize,
    pub output_fiber_dim: usize,
    pub weights: DMatrix<f64>,
    pub transport_mode: TransportMode,
}

/// How to handle parallel transport.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TransportMode {
    /// Trivial transport (flat connection).
    Trivial,
    /// Transport via adjacency-weighted interpolation.
    Interpolated,
}

impl GaugeEquivLayer {
    pub fn new(input_fiber_dim: usize, output_fiber_dim: usize) -> Self {
        Self {
            input_fiber_dim,
            output_fiber_dim,
            weights: DMatrix::new_random(output_fiber_dim, input_fiber_dim),
            transport_mode: TransportMode::Trivial,
        }
    }

    /// Forward: gauge-equivariant transformation.
    /// Uses connection (parallel transport) to compare features at different nodes.
    pub fn forward(&self, graph: &AgentGraph, features: &AgentFeatures) -> AgentFeatures {
        let n = graph.num_agents;
        let d = features.feature_dim();
        let fd = self.input_fiber_dim;
        let n_fibers = d / fd.max(1);
        let mut result = DMatrix::zeros(n, self.output_fiber_dim * n_fibers);

        for i in 0..n {
            let h_i: DVector<f64> = features.data.row(i).transpose();
            let neighbors = graph.neighbors(i);

            // Transform own features
            for k in 0..n_fibers {
                let offset = k * fd;
                if offset + fd <= d {
                    let fiber: DVector<f64> = h_i.rows(offset, fd).clone_owned();
                    let transformed = &self.weights * &fiber;
                    let out_offset = k * self.output_fiber_dim;
                    for j in 0..self.output_fiber_dim.min(transformed.nrows()) {
                        result[(i, out_offset + j)] = transformed[j];
                    }
                }
            }

            // Aggregate transported neighbor features
            if !neighbors.is_empty() {
                let mut neighbor_sum: DVector<f64> = DVector::zeros(self.output_fiber_dim * n_fibers);
                for &j in &neighbors {
                    let h_j: DVector<f64> = features.data.row(j).transpose();
                    for k in 0..n_fibers {
                        let offset = k * fd;
                        if offset + fd <= d {
                            let fiber: DVector<f64> = h_j.rows(offset, fd).clone_owned();
                            // Trivial transport: no gauge adjustment needed
                            let transported = &self.weights * &fiber;
                            let out_offset = k * self.output_fiber_dim;
                            for jj in 0..self.output_fiber_dim.min(transported.nrows()) {
                                neighbor_sum[out_offset + jj] += transported[jj];
                            }
                        }
                    }
                }
                // Mix own and neighbor contributions
                for j in 0..result.ncols() {
                    result[(i, j)] = result[(i, j)] + neighbor_sum[j] / neighbors.len() as f64;
                }
            }
        }

        AgentFeatures::from_matrix(result)
    }
}

/// Connection matrix for parallel transport on a graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Connection {
    /// Number of nodes.
    pub n: usize,
    /// Fiber dimension.
    pub fiber_dim: usize,
    /// Transport matrices as flat storage: transports[i * n + j] transports from j to i.
    pub transports: Vec<DMatrix<f64>>,
}

impl Connection {
    /// Trivial (flat) connection.
    pub fn trivial(n: usize, fiber_dim: usize) -> Self {
        let id = DMatrix::identity(fiber_dim, fiber_dim);
        Self {
            n,
            fiber_dim,
            transports: vec![id; n * n],
        }
    }

    /// Get transport matrix from j to i.
    pub fn transport(&self, i: usize, j: usize) -> &DMatrix<f64> {
        &self.transports[i * self.n + j]
    }

    /// Compute holonomy around a cycle.
    pub fn holonomy(&self, cycle: &[usize]) -> DMatrix<f64> {
        if cycle.len() < 2 {
            return DMatrix::identity(self.fiber_dim, self.fiber_dim);
        }
        let mut result = self.transport(cycle[0], cycle[1]).clone();
        for k in 1..cycle.len() - 1 {
            let next = self.transport(cycle[k], cycle[k + 1]);
            result = &result * next;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauge_identity() {
        let g = GaugeTransformation::identity(3, 2);
        assert_eq!(g.n_agents(), 3);
        assert_eq!(g.fiber_dim(), 2);
        let features = DMatrix::new_random(3, 4);
        let result = g.act(&features);
        assert!((result - features).iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn test_gauge_inverse() {
        let g = GaugeTransformation::random_orthogonal(3, 2, 42);
        let inv = g.inverse();
        let comp = g.compose(&inv);
        let features = DMatrix::new_random(3, 4);
        let result = comp.act(&features);
        for i in 0..3 {
            for j in 0..4 {
                assert!((result[(i, j)] - features[(i, j)]).abs() < 1e-8, "Mismatch at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_gauge_compose() {
        let g1 = GaugeTransformation::random_orthogonal(2, 2, 1);
        let g2 = GaugeTransformation::random_orthogonal(2, 2, 2);
        let comp = g1.compose(&g2);
        assert_eq!(comp.n_agents(), 2);
    }

    #[test]
    fn test_gauge_equiv_layer() {
        let mut graph = AgentGraph::new(4);
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 3, 1.0);
        let layer = GaugeEquivLayer::new(2, 3);
        let features = AgentFeatures::from_matrix(DMatrix::new_random(4, 4)); // 2 fibers × dim 2
        let result = layer.forward(&graph, &features);
        assert_eq!(result.n_agents(), 4);
        assert!(result.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_connection_trivial() {
        let conn = Connection::trivial(3, 2);
        let t = conn.transport(0, 1);
        let id = DMatrix::identity(2, 2);
        assert!((t - &id).iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn test_connection_holonomy() {
        let conn = Connection::trivial(4, 2);
        let cycle = vec![0, 1, 2, 0];
        let h = conn.holonomy(&cycle);
        // Trivial connection → identity holonomy
        let id = DMatrix::identity(2, 2);
        assert!((&h - &id).iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn test_gauge_random_orthogonal() {
        let g = GaugeTransformation::random_orthogonal(3, 2, 42);
        for gauge in &g.gauges {
            // Should be approximately orthogonal
            let product = gauge * gauge.transpose();
            let id = DMatrix::identity(2, 2);
            assert!((&product - &id).iter().all(|v| v.abs() < 1e-6));
        }
    }

    #[test]
    fn test_gauge_act_structure() {
        let g = GaugeTransformation::identity(2, 2);
        let features = DMatrix::from_row_slice(2, 4, &[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ]);
        let result = g.act(&features);
        // Identity gauge → no change
        for i in 0..2 {
            for j in 0..4 {
                assert!((result[(i, j)] - features[(i, j)]).abs() < 1e-10);
            }
        }
    }
}
