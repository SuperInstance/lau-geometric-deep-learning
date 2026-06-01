//! Core types and traits for geometric deep learning.

use nalgebra::{DMatrix, DVector, ComplexField};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Create a random DVector of given size.
pub fn random_vector(n: usize) -> DVector<f64> {
    DVector::from_vec((0..n).map(|i| (i as f64 * 0.12345 + 0.6789) % 1.0).collect())
}

/// A symmetry group element — can act on features.
pub trait GroupAction: Clone + Debug + Send + Sync {
    /// Apply the group action to a feature matrix (n_agents × feature_dim).
    fn act(&self, features: &DMatrix<f64>) -> DMatrix<f64>;

    /// Inverse of this group element.
    fn inverse(&self) -> Self;

    /// Compose with another group element: self ∘ other.
    fn compose(&self, other: &Self) -> Self;

    /// The identity element.
    fn identity() -> Self;
}

/// An equivariant layer: f(g·x) = g·f(x) for all g in G.
pub trait EquivariantLayer: Clone + Debug + Send + Sync {
    type Group: GroupAction;

    /// Forward pass: transform features.
    fn forward(&self, features: &DMatrix<f64>) -> DMatrix<f64>;

    /// Check equivariance for a given group element.
    fn check_equivariance(&self, g: &Self::Group, features: &DMatrix<f64>, tol: f64) -> bool {
        let gx = g.act(features);
        let f_gx = self.forward(&gx);
        let fx = self.forward(features);
        let g_fx = g.act(&fx);
        let diff = &f_gx - &g_fx;
        diff.iter().all(|v| v.abs() < tol)
    }
}

/// A symmetry group specification.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SymmetryType {
    Permutation,
    Translation { dim: usize },
    Rotation { dim: usize },
    Scale,
    Time,
    Gauge { fiber_dim: usize },
    GroupConv { order: usize },
}

/// Configuration for a geometric deep learning model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeometricModelConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub num_layers: usize,
    pub symmetries: Vec<SymmetryType>,
    pub use_spectral: bool,
    pub use_spatial: bool,
    pub num_agents: usize,
}

impl Default for GeometricModelConfig {
    fn default() -> Self {
        Self {
            input_dim: 16,
            hidden_dim: 32,
            output_dim: 8,
            num_layers: 3,
            symmetries: vec![SymmetryType::Permutation],
            use_spectral: false,
            use_spatial: true,
            num_agents: 10,
        }
    }
}

/// Graph structure for agent systems.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentGraph {
    /// Number of agents (nodes).
    pub num_agents: usize,
    /// Adjacency matrix.
    pub adjacency: DMatrix<f64>,
    /// Edge weights (flattened upper triangle of adjacency).
    pub edge_features: Vec<f64>,
}

impl AgentGraph {
    /// Create an empty graph with n agents.
    pub fn new(n: usize) -> Self {
        Self {
            num_agents: n,
            adjacency: DMatrix::zeros(n, n),
            edge_features: Vec::new(),
        }
    }

    /// Create a fully connected graph.
    pub fn fully_connected(n: usize) -> Self {
        let adj = DMatrix::from_element(n, n, 1.0);
        let edge_features = vec![1.0; n * n];
        Self { num_agents: n, adjacency: adj, edge_features }
    }

    /// Create a ring graph.
    pub fn ring(n: usize) -> Self {
        let mut adj = DMatrix::zeros(n, n);
        for i in 0..n {
            let j = (i + 1) % n;
            adj[(i, j)] = 1.0;
            adj[(j, i)] = 1.0;
        }
        Self { num_agents: n, adjacency: adj, edge_features: Vec::new() }
    }

    /// Create a k-nearest-neighbor graph from positions.
    pub fn knn(positions: &DMatrix<f64>, k: usize) -> Self {
        let n = positions.nrows();
        let dim = positions.ncols();
        let mut adj = DMatrix::zeros(n, n);
        for i in 0..n {
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let mut d = 0.0;
                    for dd in 0..dim {
                        let diff = positions[(i, dd)] - positions[(j, dd)];
                        d += diff * diff;
                    }
                    (j, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for (j, _) in dists.iter().take(k) {
                adj[(i, *j)] = 1.0;
                adj[(*j, i)] = 1.0;
            }
        }
        Self { num_agents: n, adjacency: adj, edge_features: Vec::new() }
    }

    /// Degree matrix (diagonal).
    pub fn degree_matrix(&self) -> DMatrix<f64> {
        let mut deg = DMatrix::zeros(self.num_agents, self.num_agents);
        for i in 0..self.num_agents {
            let d: f64 = (0..self.num_agents).map(|j| self.adjacency[(i, j)]).sum();
            deg[(i, i)] = d;
        }
        deg
    }

    /// Laplacian L = D - A.
    pub fn laplacian(&self) -> DMatrix<f64> {
        &self.degree_matrix() - &self.adjacency
    }

    /// Normalized Laplacian L_norm = I - D^{-1/2} A D^{-1/2}.
    pub fn normalized_laplacian(&self) -> DMatrix<f64> {
        let deg = self.degree_matrix();
        let n = self.num_agents;
        let mut d_inv_sqrt = DMatrix::zeros(n, n);
        for i in 0..n {
            if deg[(i, i)] > 0.0 {
                d_inv_sqrt[(i, i)] = 1.0 / deg[(i, i)].sqrt();
            }
        }
        let identity = DMatrix::identity(n, n);
        &identity - &d_inv_sqrt * &self.adjacency * &d_inv_sqrt
    }

    /// Add an edge.
    pub fn add_edge(&mut self, i: usize, j: usize, weight: f64) {
        assert!(i < self.num_agents && j < self.num_agents, "Edge indices out of bounds");
        self.adjacency[(i, j)] = weight;
        self.adjacency[(j, i)] = weight;
    }

    /// Number of edges (counting each undirected edge once).
    pub fn num_edges(&self) -> usize {
        let mut count = 0;
        for i in 0..self.num_agents {
            for j in i..self.num_agents {
                if self.adjacency[(i, j)] > 0.0 {
                    count += 1;
                }
            }
        }
        count
    }

    /// Check if graph is connected.
    pub fn is_connected(&self) -> bool {
        if self.num_agents == 0 {
            return true;
        }
        let mut visited = vec![false; self.num_agents];
        let mut stack = vec![0];
        visited[0] = true;
        while let Some(node) = stack.pop() {
            for j in 0..self.num_agents {
                if self.adjacency[(node, j)] > 0.0 && !visited[j] {
                    visited[j] = true;
                    stack.push(j);
                }
            }
        }
        visited.iter().all(|&v| v)
    }

    /// Get neighbors of node i.
    pub fn neighbors(&self, i: usize) -> Vec<usize> {
        (0..self.num_agents)
            .filter(|&j| self.adjacency[(i, j)] > 0.0)
            .collect()
    }
}

/// Feature tensor for agent systems: (n_agents × feature_dim).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentFeatures {
    pub data: DMatrix<f64>,
}

impl AgentFeatures {
    pub fn new(n_agents: usize, feature_dim: usize) -> Self {
        Self { data: DMatrix::zeros(n_agents, feature_dim) }
    }

    pub fn from_matrix(data: DMatrix<f64>) -> Self {
        Self { data }
    }

    pub fn n_agents(&self) -> usize {
        self.data.nrows()
    }

    pub fn feature_dim(&self) -> usize {
        self.data.ncols()
    }

    pub fn random(n_agents: usize, feature_dim: usize) -> Self {
        Self { data: DMatrix::new_random(n_agents, feature_dim) }
    }

    /// Permute agents by a permutation index.
    pub fn permute(&self, perm: &[usize]) -> Self {
        assert_eq!(perm.len(), self.n_agents());
        let n = self.n_agents();
        let d = self.feature_dim();
        let mut result = DMatrix::zeros(n, d);
        for (new_idx, &old_idx) in perm.iter().enumerate() {
            for j in 0..d {
                result[(new_idx, j)] = self.data[(old_idx, j)];
            }
        }
        Self { data: result }
    }

    /// Translate features by a shift vector.
    pub fn translate(&self, shift: &DVector<f64>) -> Self {
        let mut result = self.data.clone();
        for i in 0..self.n_agents() {
            for j in 0..self.feature_dim().min(shift.nrows()) {
                result[(i, j)] += shift[j];
            }
        }
        Self { data: result }
    }

    /// Scale features by a scalar.
    pub fn scale(&self, s: f64) -> Self {
        Self { data: &self.data * s }
    }

    /// L2 norm of features across all agents.
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_new() {
        let g = AgentGraph::new(5);
        assert_eq!(g.num_agents, 5);
        assert_eq!(g.adjacency.nrows(), 5);
        assert!(g.adjacency.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_graph_fully_connected() {
        let g = AgentGraph::fully_connected(3);
        assert_eq!(g.adjacency[(0, 1)], 1.0);
        assert_eq!(g.adjacency[(1, 2)], 1.0);
        assert_eq!(g.adjacency[(0, 0)], 1.0);
    }

    #[test]
    fn test_graph_ring() {
        let g = AgentGraph::ring(4);
        assert_eq!(g.adjacency[(0, 1)], 1.0);
        assert_eq!(g.adjacency[(1, 2)], 1.0);
        assert_eq!(g.adjacency[(2, 3)], 1.0);
        assert_eq!(g.adjacency[(3, 0)], 1.0);
        assert_eq!(g.adjacency[(0, 2)], 0.0);
    }

    #[test]
    fn test_graph_add_edge() {
        let mut g = AgentGraph::new(4);
        g.add_edge(0, 1, 1.0);
        assert_eq!(g.adjacency[(0, 1)], 1.0);
        assert_eq!(g.adjacency[(1, 0)], 1.0);
    }

    #[test]
    fn test_graph_degree_matrix() {
        let mut g = AgentGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let deg = g.degree_matrix();
        assert_eq!(deg[(0, 0)], 1.0);
        assert_eq!(deg[(1, 1)], 2.0);
        assert_eq!(deg[(2, 2)], 1.0);
    }

    #[test]
    fn test_graph_laplacian() {
        let mut g = AgentGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let lap = g.laplacian();
        // Row 0: [1, -1, 0]
        assert!((lap[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((lap[(0, 1)] - (-1.0)).abs() < 1e-10);
        assert!((lap[(0, 2)] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_graph_normalized_laplacian() {
        let g = AgentGraph::fully_connected(3);
        let l_norm = g.normalized_laplacian();
        // For fully connected, diagonal should be 1 - 1/n = 2/3
        assert!((l_norm[(0, 0)] - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_graph_knn() {
        let pos = DMatrix::from_row_slice(4, 2, &[
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            5.0, 5.0,
        ]);
        let g = AgentGraph::knn(&pos, 2);
        assert_eq!(g.num_agents, 4);
        // Node 3 (5,5) should be far from others
        assert_eq!(g.adjacency[(0, 3)], 0.0);
    }

    #[test]
    fn test_graph_is_connected() {
        let g = AgentGraph::ring(5);
        assert!(g.is_connected());
        let mut g2 = AgentGraph::new(4);
        g2.add_edge(0, 1, 1.0);
        g2.add_edge(2, 3, 1.0);
        assert!(!g2.is_connected());
    }

    #[test]
    fn test_graph_neighbors() {
        let mut g = AgentGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(0, 2, 2.0);
        let nbrs = g.neighbors(0);
        assert_eq!(nbrs.len(), 2);
        assert!(nbrs.contains(&1));
        assert!(nbrs.contains(&2));
    }

    #[test]
    fn test_graph_num_edges() {
        let mut g = AgentGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(2, 3, 1.0);
        assert_eq!(g.num_edges(), 2);
    }

    #[test]
    fn test_agent_features_new() {
        let f = AgentFeatures::new(5, 3);
        assert_eq!(f.n_agents(), 5);
        assert_eq!(f.feature_dim(), 3);
    }

    #[test]
    fn test_agent_features_permute() {
        let f = AgentFeatures::from_matrix(DMatrix::from_row_slice(3, 2, &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]));
        let permuted = f.permute(&[2, 0, 1]);
        assert!((permuted.data[(0, 0)] - 5.0).abs() < 1e-10);
        assert!((permuted.data[(1, 0)] - 1.0).abs() < 1e-10);
        assert!((permuted.data[(2, 0)] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_agent_features_translate() {
        let f = AgentFeatures::from_matrix(DMatrix::from_row_slice(2, 2, &[
            1.0, 2.0,
            3.0, 4.0,
        ]));
        let shift = DVector::from_vec(vec![10.0, 20.0]);
        let translated = f.translate(&shift);
        assert!((translated.data[(0, 0)] - 11.0).abs() < 1e-10);
        assert!((translated.data[(0, 1)] - 22.0).abs() < 1e-10);
    }

    #[test]
    fn test_agent_features_scale() {
        let f = AgentFeatures::from_matrix(DMatrix::from_row_slice(2, 2, &[
            1.0, 2.0,
            3.0, 4.0,
        ]));
        let scaled = f.scale(2.0);
        assert!((scaled.data[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((scaled.data[(1, 1)] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_agent_features_norm() {
        let f = AgentFeatures::from_matrix(DMatrix::from_row_slice(2, 1, &[3.0, 4.0]));
        assert!((f.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_model_config_default() {
        let config = GeometricModelConfig::default();
        assert_eq!(config.input_dim, 16);
        assert_eq!(config.num_layers, 3);
        assert_eq!(config.symmetries.len(), 1);
    }

    #[test]
    fn test_model_config_serialize() {
        let config = GeometricModelConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let config2: GeometricModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.input_dim, config2.input_dim);
    }

    #[test]
    fn test_symmetry_type_serialize() {
        let s = SymmetryType::Translation { dim: 3 };
        let json = serde_json::to_string(&s).unwrap();
        let s2: SymmetryType = serde_json::from_str(&json).unwrap();
        assert_eq!(s, s2);
    }
}
