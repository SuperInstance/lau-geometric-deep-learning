//! Universal approximation for equivariant networks.
//!
//! Theoretical results showing that equivariant networks can approximate
//! any continuous equivariant function.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::core::AgentFeatures;
use crate::permutation::{Aggregation, DeepSetsLayer};

/// Universal approximation theorem for permutation-equivariant functions.
///
/// Any continuous permutation-equivariant function can be approximated
/// by a Deep Sets architecture with sufficient hidden width.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UniversalApproximator {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub depth: usize,
    pub layers: Vec<DeepSetsLayer>,
}

impl UniversalApproximator {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, depth: usize) -> Self {
        let mut layers = Vec::new();
        layers.push(DeepSetsLayer::new(input_dim, hidden_dim, Aggregation::Sum));
        for _ in 1..depth.saturating_sub(1) {
            layers.push(DeepSetsLayer::new(hidden_dim, hidden_dim, Aggregation::Sum));
        }
        if depth > 1 {
            layers.push(DeepSetsLayer::new(hidden_dim, output_dim, Aggregation::Sum));
        }
        Self { input_dim, hidden_dim, output_dim, depth, layers }
    }

    /// Forward through all layers.
    pub fn forward(&self, features: &AgentFeatures) -> AgentFeatures {
        let mut current = features.clone();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }

    /// Estimate approximation error for a given function.
    /// Uses empirical error between network output and target.
    pub fn empirical_error(
        &self,
        features: &AgentFeatures,
        target: &AgentFeatures,
    ) -> f64 {
        let output = self.forward(features);
        let n = output.n_agents();
        let d = output.feature_dim();
        let mut sum_sq = 0.0;
        for i in 0..n {
            for j in 0..d.min(target.feature_dim()) {
                let diff = output.data[(i, j)] - target.data[(i, j)];
                sum_sq += diff * diff;
            }
        }
        sum_sq.sqrt()
    }
}

/// Approximation capacity metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApproximationCapacity {
    pub hidden_dim: usize,
    pub depth: usize,
    pub num_parameters: usize,
    pub theoretical_bound: f64,
}

impl ApproximationCapacity {
    /// Compute capacity of a permutation-equivariant universal approximator.
    pub fn for_permutation_equiv(input_dim: usize, hidden_dim: usize, output_dim: usize, depth: usize) -> Self {
        // Count parameters: each DeepSetsLayer has phi_weights + phi_bias + rho_weights + rho_bias
        let hidden = (input_dim + output_dim) / 2 + 1;
        let params_per_layer = hidden * input_dim + hidden + output_dim * hidden + output_dim;
        let num_params = params_per_layer * depth;
        Self {
            hidden_dim,
            depth,
            num_parameters: num_params,
            theoretical_bound: 1.0 / (hidden_dim as f64).sqrt(), // Rough bound
        }
    }
}

/// Steerable neural network: a framework for universal equivariant approximation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SteerableNetwork {
    pub input_type: IrrepType,
    pub output_type: IrrepType,
    pub hidden_multiplicities: Vec<usize>,
    pub weights: Vec<DMatrix<f64>>,
}

/// Irreducible representation type.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum IrrepType {
    /// Scalar (trivial representation).
    Scalar,
    /// Vector (standard representation).
    Vector { dim: usize },
    /// Regular representation of a group.
    Regular { group_order: usize },
    /// Direct sum of irreps.
    DirectSum { types: Vec<IrrepType> },
}

impl SteerableNetwork {
    pub fn new(input_type: IrrepType, output_type: IrrepType, hidden_multiplicities: Vec<usize>) -> Self {
        let input_dim = input_type.dimension();
        let output_dim = output_type.dimension();
        let mut weights = Vec::new();

        let mut prev_dim = input_dim;
        for &mult in &hidden_multiplicities {
            weights.push(DMatrix::new_random(mult, prev_dim));
            prev_dim = mult;
        }
        weights.push(DMatrix::new_random(output_dim, prev_dim));

        Self { input_type, output_type, hidden_multiplicities, weights }
    }

    /// Forward pass.
    pub fn forward(&self, input: &DVector<f64>) -> DVector<f64> {
        let mut current = input.clone();
        for w in &self.weights {
            current = w * &current;
            // ReLU
            current = current.map(|v| if v > 0.0 { v } else { 0.0 });
        }
        current
    }

    /// Forward on agent features.
    pub fn forward_features(&self, features: &AgentFeatures) -> AgentFeatures {
        let n = features.n_agents();
        let _d = features.feature_dim();
        let out_dim = self.output_type.dimension();
        let mut result = DMatrix::zeros(n, out_dim);
        for i in 0..n {
            let row: DVector<f64> = features.data.row(i).transpose();
            let out = self.forward(&row);
            for j in 0..out_dim.min(out.nrows()) {
                result[(i, j)] = out[j];
            }
        }
        AgentFeatures::from_matrix(result)
    }
}

impl IrrepType {
    pub fn dimension(&self) -> usize {
        match self {
            IrrepType::Scalar => 1,
            IrrepType::Vector { dim } => *dim,
            IrrepType::Regular { group_order } => *group_order,
            IrrepType::DirectSum { types } => types.iter().map(|t| t.dimension()).sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_universal_approximator_new() {
        let ua = UniversalApproximator::new(4, 8, 3, 3);
        assert_eq!(ua.depth, 3);
        assert_eq!(ua.layers.len(), 3);
    }

    #[test]
    fn test_universal_approximator_forward() {
        let ua = UniversalApproximator::new(4, 8, 3, 2);
        let features = AgentFeatures::random(5, 4);
        let result = ua.forward(&features);
        assert_eq!(result.n_agents(), 5);
        assert_eq!(result.feature_dim(), 3);
    }

    #[test]
    fn test_universal_approximator_single_layer() {
        let ua = UniversalApproximator::new(3, 8, 2, 1);
        assert_eq!(ua.layers.len(), 1);
        let features = AgentFeatures::random(4, 3);
        let result = ua.forward(&features);
        assert_eq!(result.n_agents(), 4);
    }

    #[test]
    fn test_empirical_error() {
        let ua = UniversalApproximator::new(3, 16, 3, 3);
        let features = AgentFeatures::random(4, 3);
        let target = AgentFeatures::random(4, 3);
        let error = ua.empirical_error(&features, &target);
        assert!(error >= 0.0);
        assert!(error.is_finite());
    }

    #[test]
    fn test_approximation_capacity() {
        let cap = ApproximationCapacity::for_permutation_equiv(4, 32, 3, 3);
        assert!(cap.num_parameters > 0);
        assert!(cap.theoretical_bound > 0.0);
    }

    #[test]
    fn test_irrep_scalar() {
        let t = IrrepType::Scalar;
        assert_eq!(t.dimension(), 1);
    }

    #[test]
    fn test_irrep_vector() {
        let t = IrrepType::Vector { dim: 3 };
        assert_eq!(t.dimension(), 3);
    }

    #[test]
    fn test_irrep_regular() {
        let t = IrrepType::Regular { group_order: 6 };
        assert_eq!(t.dimension(), 6);
    }

    #[test]
    fn test_irrep_direct_sum() {
        let t = IrrepType::DirectSum {
            types: vec![IrrepType::Scalar, IrrepType::Vector { dim: 3 }],
        };
        assert_eq!(t.dimension(), 4);
    }

    #[test]
    fn test_steerable_network() {
        let net = SteerableNetwork::new(
            IrrepType::Vector { dim: 4 },
            IrrepType::Vector { dim: 2 },
            vec![8, 6],
        );
        let input = crate::core::random_vector(4);
        let output = net.forward(&input);
        assert_eq!(output.nrows(), 2);
    }

    #[test]
    fn test_steerable_network_features() {
        let net = SteerableNetwork::new(
            IrrepType::Scalar,
            IrrepType::Scalar,
            vec![4, 4],
        );
        let features = AgentFeatures::random(3, 1);
        let result = net.forward_features(&features);
        assert_eq!(result.n_agents(), 3);
        assert_eq!(result.feature_dim(), 1);
    }

    #[test]
    fn test_universal_approximator_deep() {
        let ua = UniversalApproximator::new(4, 16, 2, 5);
        assert_eq!(ua.layers.len(), 5);
        let features = AgentFeatures::random(3, 4);
        let result = ua.forward(&features);
        assert_eq!(result.feature_dim(), 2);
    }
}
