//! Spatial filters: message passing with geometric weight sharing.
//!
//! Message passing neural networks (MPNNs) that aggregate information
//! from neighboring agents on a graph.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::core::{AgentFeatures, AgentGraph};
use crate::core::random_vector;

/// Message passing layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MessagePassingLayer {
    pub input_dim: usize,
    pub output_dim: usize,
    pub message_weights: DMatrix<f64>,
    pub update_weights: DMatrix<f64>,
    pub aggregation: MessageAggregation,
}

/// How to aggregate messages from neighbors.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum MessageAggregation {
    Sum,
    Mean,
    Max,
}

impl MessagePassingLayer {
    pub fn new(input_dim: usize, output_dim: usize, aggregation: MessageAggregation) -> Self {
        Self {
            input_dim,
            output_dim,
            message_weights: DMatrix::new_random(output_dim, input_dim * 2),
            update_weights: DMatrix::new_random(output_dim, input_dim + output_dim),
            aggregation,
        }
    }

    /// Compute message from node i to node j.
    pub fn message(&self, h_i: &DVector<f64>, h_j: &DVector<f64>) -> DVector<f64> {
        let combined = DVector::from_vec(
            h_i.iter().chain(h_j.iter()).copied().collect(),
        );
        let mut msg = &self.message_weights * &combined;
        // ReLU
        for v in msg.iter_mut() {
            if *v < 0.0 { *v = 0.0; }
        }
        msg
    }

    /// Aggregate messages from neighbors.
    pub fn aggregate(&self, messages: &[DVector<f64>]) -> DVector<f64> {
        if messages.is_empty() {
            return DVector::zeros(self.output_dim);
        }
        match self.aggregation {
            MessageAggregation::Sum => {
                messages.iter().fold(DVector::zeros(messages[0].nrows()), |acc, m| acc + m)
            }
            MessageAggregation::Mean => {
                let sum = messages.iter().fold(DVector::zeros(messages[0].nrows()), |acc, m| acc + m);
                &sum / messages.len() as f64
            }
            MessageAggregation::Max => {
                let dim = messages[0].nrows();
                let mut result = messages[0].clone();
                for m in messages.iter().skip(1) {
                    for i in 0..dim {
                        result[i] = result[i].max(m[i]);
                    }
                }
                result
            }
        }
    }

    /// Update node feature given old feature and aggregated message.
    pub fn update(&self, h: &DVector<f64>, m: &DVector<f64>) -> DVector<f64> {
        let combined = DVector::from_vec(
            h.iter().chain(m.iter()).copied().collect(),
        );
        let mut out = &self.update_weights * &combined;
        for v in out.iter_mut() {
            if *v < 0.0 { *v = 0.0; }
        }
        out
    }

    /// Forward: one round of message passing.
    pub fn forward(&self, graph: &AgentGraph, features: &AgentFeatures) -> AgentFeatures {
        let n = graph.num_agents;
        let mut result = DMatrix::zeros(n, self.output_dim);

        for i in 0..n {
            let h_i: DVector<f64> = features.data.row(i).transpose();
            let neighbors = graph.neighbors(i);

            let messages: Vec<DVector<f64>> = neighbors.iter().map(|&j| {
                let h_j: DVector<f64> = features.data.row(j).transpose();
                self.message(&h_i, &h_j)
            }).collect();

            let agg = self.aggregate(&messages);
            let updated = self.update(&h_i, &agg);

            for j in 0..self.output_dim.min(updated.nrows()) {
                result[(i, j)] = updated[j];
            }
        }

        AgentFeatures::from_matrix(result)
    }
}

/// Graph convolution layer (simplified GAT-like attention).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphConvLayer {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: DMatrix<f64>,
    pub attention_weights: DVector<f64>,
    pub use_attention: bool,
}

impl GraphConvLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            weights: DMatrix::new_random(output_dim, input_dim),
            attention_weights: random_vector(2 * input_dim),
            use_attention: false,
        }
    }

    pub fn with_attention(mut self) -> Self {
        self.use_attention = true;
        self
    }

    /// Compute attention coefficient between nodes i and j.
    pub fn attention(&self, h_i: &DVector<f64>, h_j: &DVector<f64>) -> f64 {
        let combined = DVector::from_vec(
            h_i.iter().chain(h_j.iter()).copied().collect(),
        );
        let dim = combined.nrows().min(self.attention_weights.nrows());
        let dot: f64 = (0..dim).map(|k| combined[k] * self.attention_weights[k]).sum();
        dot.leaky_relu(0.2)
    }

    /// Forward pass.
    pub fn forward(&self, graph: &AgentGraph, features: &AgentFeatures) -> AgentFeatures {
        let n = graph.num_agents;
        let mut result = DMatrix::zeros(n, self.output_dim);

        for i in 0..n {
            let h_i: DVector<f64> = features.data.row(i).transpose();
            let neighbors = graph.neighbors(i);

            if neighbors.is_empty() {
                let out = &self.weights * &h_i;
                for j in 0..self.output_dim.min(out.nrows()) {
                    result[(i, j)] = out[j];
                }
                continue;
            }

            if self.use_attention {
                // Attention-weighted aggregation
                let attn_coeffs: Vec<f64> = neighbors.iter().map(|&j| {
                    let h_j: DVector<f64> = features.data.row(j).transpose();
                    self.attention(&h_i, &h_j).exp()
                }).collect();
                let attn_sum: f64 = attn_coeffs.iter().sum();

                let mut weighted_sum = DVector::zeros(self.input_dim);
                for (&j, &a) in neighbors.iter().zip(attn_coeffs.iter()) {
                    let h_j: DVector<f64> = features.data.row(j).transpose();
                    weighted_sum += (a / attn_sum) * &h_j;
                }
                let out = &self.weights * &(&weighted_sum + &h_i);
                for j in 0..self.output_dim.min(out.nrows()) {
                    result[(i, j)] = out[j];
                }
            } else {
                // Simple mean aggregation
                let mut sum = h_i.clone();
                for &j in &neighbors {
                    let h_j: DVector<f64> = features.data.row(j).transpose();
                    sum += h_j;
                }
                sum = &sum / (neighbors.len() + 1) as f64;
                let out = &self.weights * &sum;
                for j in 0..self.output_dim.min(out.nrows()) {
                    result[(i, j)] = out[j];
                }
            }
        }

        AgentFeatures::from_matrix(result)
    }
}

/// Multi-layer GNN stack.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GNNScheme {
    pub layers: Vec<MessagePassingLayer>,
}

impl GNNScheme {
    pub fn new(layer_dims: &[usize], aggregation: MessageAggregation) -> Self {
        let layers = layer_dims.windows(2).map(|w| {
            MessagePassingLayer::new(w[0], w[1], aggregation.clone())
        }).collect();
        Self { layers }
    }

    /// Forward through all layers.
    pub fn forward(&self, graph: &AgentGraph, features: &AgentFeatures) -> AgentFeatures {
        let mut current = features.clone();
        for layer in &self.layers {
            current = layer.forward(graph, &current);
        }
        current
    }

    /// Readout: aggregate all agent features into a single vector.
    pub fn readout_sum(&self, features: &AgentFeatures) -> DVector<f64> {
        let d = features.feature_dim();
        let mut sum = DVector::zeros(d);
        for i in 0..features.n_agents() {
            for j in 0..d {
                sum[j] += features.data[(i, j)];
            }
        }
        sum
    }

    /// Readout with attention.
    pub fn readout_attention(&self, features: &AgentFeatures) -> DVector<f64> {
        let d = features.feature_dim();
        // Simple attention: weight by L2 norm of each agent's features
        let norms: Vec<f64> = (0..features.n_agents()).map(|i| {
            (0..d).map(|j| features.data[(i, j)] * features.data[(i, j)]).sum::<f64>().sqrt()
        }).collect();
        let total: f64 = norms.iter().sum();

        let mut result = DVector::zeros(d);
        for i in 0..features.n_agents() {
            let w = if total > 1e-10 { norms[i] / total } else { 1.0 / features.n_agents() as f64 };
            for j in 0..d {
                result[j] += w * features.data[(i, j)];
            }
        }
        result
    }
}

trait LeakyReLU {
    fn leaky_relu(&self, alpha: f64) -> f64;
}

impl LeakyReLU for f64 {
    fn leaky_relu(&self, alpha: f64) -> f64 {
        if *self > 0.0 { *self } else { alpha * self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph() -> AgentGraph {
        let mut g = AgentGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        g
    }

    #[test]
    fn test_message_passing_forward() {
        let graph = make_graph();
        let features = AgentFeatures::from_matrix(DMatrix::new_random(4, 3));
        let layer = MessagePassingLayer::new(3, 4, MessageAggregation::Sum);
        let result = layer.forward(&graph, &features);
        assert_eq!(result.n_agents(), 4);
        assert_eq!(result.feature_dim(), 4);
    }

    #[test]
    fn test_message_passing_mean() {
        let graph = make_graph();
        let features = AgentFeatures::from_matrix(DMatrix::new_random(4, 3));
        let layer = MessagePassingLayer::new(3, 4, MessageAggregation::Mean);
        let result = layer.forward(&graph, &features);
        assert_eq!(result.n_agents(), 4);
        assert!(result.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_message_passing_max() {
        let graph = make_graph();
        let features = AgentFeatures::from_matrix(DMatrix::new_random(4, 3));
        let layer = MessagePassingLayer::new(3, 4, MessageAggregation::Max);
        let result = layer.forward(&graph, &features);
        assert_eq!(result.n_agents(), 4);
    }

    #[test]
    fn test_graph_conv_forward() {
        let graph = make_graph();
        let features = AgentFeatures::from_matrix(DMatrix::new_random(4, 3));
        let layer = GraphConvLayer::new(3, 4);
        let result = layer.forward(&graph, &features);
        assert_eq!(result.n_agents(), 4);
        assert_eq!(result.feature_dim(), 4);
    }

    #[test]
    fn test_graph_conv_attention() {
        let graph = make_graph();
        let features = AgentFeatures::from_matrix(DMatrix::new_random(4, 3));
        let layer = GraphConvLayer::new(3, 4).with_attention();
        let result = layer.forward(&graph, &features);
        assert_eq!(result.n_agents(), 4);
        assert!(result.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_gnn_scheme() {
        let graph = make_graph();
        let features = AgentFeatures::from_matrix(DMatrix::new_random(4, 3));
        let gnn = GNNScheme::new(&[3, 5, 4, 2], MessageAggregation::Sum);
        let result = gnn.forward(&graph, &features);
        assert_eq!(result.n_agents(), 4);
        assert_eq!(result.feature_dim(), 2);
    }

    #[test]
    fn test_gnn_readout_sum() {
        let gnn = GNNScheme::new(&[3, 5], MessageAggregation::Sum);
        let features = AgentFeatures::from_matrix(DMatrix::from_row_slice(3, 2, &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]));
        let readout = gnn.readout_sum(&features);
        assert!((readout[0] - 9.0).abs() < 1e-10);
        assert!((readout[1] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_gnn_readout_attention() {
        let gnn = GNNScheme::new(&[3, 5], MessageAggregation::Sum);
        let features = AgentFeatures::from_matrix(DMatrix::from_row_slice(3, 2, &[
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
        ]));
        let readout = gnn.readout_attention(&features);
        assert_eq!(readout.nrows(), 2);
    }

    #[test]
    fn test_isolated_node() {
        let mut graph = AgentGraph::new(3);
        graph.add_edge(0, 1, 1.0);
        // Node 2 is isolated
        let features = AgentFeatures::from_matrix(DMatrix::new_random(3, 2));
        let layer = MessagePassingLayer::new(2, 3, MessageAggregation::Sum);
        let result = layer.forward(&graph, &features);
        assert_eq!(result.n_agents(), 3);
    }

    #[test]
    fn test_message_function() {
        let layer = MessagePassingLayer::new(3, 4, MessageAggregation::Sum);
        let h_i = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let h_j = DVector::from_vec(vec![4.0, 5.0, 6.0]);
        let msg = layer.message(&h_i, &h_j);
        assert_eq!(msg.nrows(), 4);
    }
}
