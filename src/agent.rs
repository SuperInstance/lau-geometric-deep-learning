//! Agent feature extraction that respects the symmetries of the problem.
//!
//! Combines all equivariant layers into a unified framework for
//! processing agent systems.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::core::{AgentFeatures, AgentGraph, GeometricModelConfig, GroupAction, SymmetryType};
use crate::permutation::{Aggregation, DeepSetsLayer, Permutation};
use crate::rotation::RotationInvariantFeatures;
use crate::scale::{ScaleEquivLayer, ScaleMode};
use crate::spectral::SpectralFilter;
use crate::spatial::{MessageAggregation, MessagePassingLayer};
use crate::gauge::GaugeEquivLayer;

/// A complete equivariant model for agent systems.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EquivariantAgentModel {
    pub config: GeometricModelConfig,
    pub permutation_layers: Vec<DeepSetsLayer>,
    pub message_passing_layers: Vec<MessagePassingLayer>,
    pub spectral_filters: Vec<SpectralFilter>,
    pub scale_layer: Option<ScaleEquivLayer>,
    pub rotation_features: Option<RotationInvariantFeatures>,
    pub gauge_layer: Option<GaugeEquivLayer>,
}

impl EquivariantAgentModel {
    pub fn new(config: GeometricModelConfig) -> Self {
        let mut perm_layers = Vec::new();
        let mut mp_layers = Vec::new();
        let mut spec_filters = Vec::new();

        let mut current_dim = config.input_dim;
        for layer_idx in 0..config.num_layers {
            let next_dim = if layer_idx < config.num_layers - 1 {
                config.hidden_dim
            } else {
                config.hidden_dim // keep consistent
            };

            if config.symmetries.contains(&SymmetryType::Permutation) && config.use_spatial {
                // Use perm layer
                perm_layers.push(DeepSetsLayer::new(current_dim, next_dim, Aggregation::Sum));
                current_dim = next_dim;
            } else if config.use_spatial {
                mp_layers.push(MessagePassingLayer::new(current_dim, next_dim, MessageAggregation::Sum));
                current_dim = next_dim;
            } else if config.symmetries.contains(&SymmetryType::Permutation) {
                perm_layers.push(DeepSetsLayer::new(current_dim, next_dim, Aggregation::Sum));
                current_dim = next_dim;
            }
            if config.use_spectral {
                spec_filters.push(SpectralFilter::new(3));
            }
        }

        let scale_layer = if config.symmetries.contains(&SymmetryType::Scale) {
            Some(ScaleEquivLayer::new(current_dim, config.output_dim, ScaleMode::Normalize))
        } else {
            None
        };

        let rotation_features = if config.symmetries.contains(&SymmetryType::Rotation { dim: 2 }) {
            Some(RotationInvariantFeatures::default())
        } else {
            None
        };

        let gauge_layer = if config.symmetries.contains(&SymmetryType::Gauge { fiber_dim: 2 }) {
            Some(GaugeEquivLayer::new(2, config.output_dim))
        } else {
            None
        };

        Self {
            config,
            permutation_layers: perm_layers,
            message_passing_layers: mp_layers,
            spectral_filters: spec_filters,
            scale_layer,
            rotation_features,
            gauge_layer,
        }
    }

    /// Forward pass through the full model.
    pub fn forward(&self, graph: &AgentGraph, features: &AgentFeatures) -> AgentFeatures {
        let n_layers = self.permutation_layers.len().max(self.message_passing_layers.len());
        let mut current = features.clone();

        for i in 0..n_layers {
            if i < self.permutation_layers.len() {
                current = self.permutation_layers[i].forward(&current);
            }
            if i < self.message_passing_layers.len() {
                current = self.message_passing_layers[i].forward(graph, &current);
            }
        }

        // Apply scale normalization
        if let Some(ref scale) = self.scale_layer {
            current = scale.forward(&current);
        }

        // Apply gauge equivariant layer
        if let Some(ref gauge) = self.gauge_layer {
            current = gauge.forward(graph, &current);
        }

        current
    }

    /// Extract features respecting all configured symmetries.
    pub fn extract_features(&self, graph: &AgentGraph, features: &AgentFeatures) -> DMatrix<f64> {
        let result = self.forward(graph, features);
        result.data
    }

    /// Check equivariance for all configured symmetries.
    pub fn verify_equivariance(&self, graph: &AgentGraph, features: &AgentFeatures, tol: f64) -> Vec<(String, bool)> {
        let mut results = Vec::new();
        let output = self.forward(graph, features);

        // Check permutation equivariance
        if self.config.symmetries.contains(&SymmetryType::Permutation) {
            let perm = Permutation::from_seed(features.n_agents(), 42);
            let perm_features = AgentFeatures::from_matrix(perm.act(&features.data));
            let perm_output = self.forward(graph, &perm_features);
            let expected = perm.act(&output.data);
            let equiv = (&perm_output.data - &expected).iter().all(|v| v.abs() < tol);
            results.push(("permutation".to_string(), equiv));
        }

        results
    }
}

/// Builder for equivariant models.
#[derive(Clone, Debug)]
pub struct EquivariantModelBuilder {
    config: GeometricModelConfig,
}

impl EquivariantModelBuilder {
    pub fn new() -> Self {
        Self {
            config: GeometricModelConfig::default(),
        }
    }

    pub fn input_dim(mut self, dim: usize) -> Self {
        self.config.input_dim = dim;
        self
    }

    pub fn hidden_dim(mut self, dim: usize) -> Self {
        self.config.hidden_dim = dim;
        self
    }

    pub fn output_dim(mut self, dim: usize) -> Self {
        self.config.output_dim = dim;
        self
    }

    pub fn num_layers(mut self, n: usize) -> Self {
        self.config.num_layers = n;
        self
    }

    pub fn num_agents(mut self, n: usize) -> Self {
        self.config.num_agents = n;
        self
    }

    pub fn add_symmetry(mut self, sym: SymmetryType) -> Self {
        if !self.config.symmetries.contains(&sym) {
            self.config.symmetries.push(sym);
        }
        self
    }

    pub fn use_spectral(mut self, v: bool) -> Self {
        self.config.use_spectral = v;
        self
    }

    pub fn use_spatial(mut self, v: bool) -> Self {
        self.config.use_spatial = v;
        self
    }

    pub fn build(self) -> EquivariantAgentModel {
        EquivariantAgentModel::new(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_graph() -> AgentGraph {
        AgentGraph::ring(5)
    }

    #[test]
    fn test_model_new() {
        let config = GeometricModelConfig::default();
        let model = EquivariantAgentModel::new(config);
        assert!(!model.permutation_layers.is_empty());
    }

    #[test]
    fn test_model_forward() {
        let config = GeometricModelConfig {
            num_agents: 5,
            ..Default::default()
        };
        let model = EquivariantAgentModel::new(config);
        let graph = make_test_graph();
        let features = AgentFeatures::new(5, 16);
        let result = model.forward(&graph, &features);
        assert_eq!(result.n_agents(), 5);
    }

    #[test]
    fn test_model_extract_features() {
        let config = GeometricModelConfig {
            num_agents: 5,
            input_dim: 8,
            hidden_dim: 16,
            output_dim: 4,
            num_layers: 2,
            ..Default::default()
        };
        let model = EquivariantAgentModel::new(config);
        let graph = make_test_graph();
        let features = AgentFeatures::random(5, 8);
        let result = model.extract_features(&graph, &features);
        assert_eq!(result.nrows(), 5);
    }

    #[test]
    fn test_model_with_scale() {
        let config = GeometricModelConfig {
            symmetries: vec![SymmetryType::Permutation, SymmetryType::Scale],
            ..Default::default()
        };
        let model = EquivariantAgentModel::new(config);
        assert!(model.scale_layer.is_some());
    }

    #[test]
    fn test_model_with_rotation() {
        let config = GeometricModelConfig {
            symmetries: vec![SymmetryType::Rotation { dim: 2 }],
            ..Default::default()
        };
        let model = EquivariantAgentModel::new(config);
        assert!(model.rotation_features.is_some());
    }

    #[test]
    fn test_model_with_gauge() {
        let config = GeometricModelConfig {
            symmetries: vec![SymmetryType::Gauge { fiber_dim: 2 }],
            ..Default::default()
        };
        let model = EquivariantAgentModel::new(config);
        assert!(model.gauge_layer.is_some());
    }

    #[test]
    fn test_builder() {
        let model = EquivariantModelBuilder::new()
            .input_dim(8)
            .hidden_dim(16)
            .output_dim(4)
            .num_layers(2)
            .num_agents(5)
            .add_symmetry(SymmetryType::Permutation)
            .add_symmetry(SymmetryType::Scale)
            .use_spatial(true)
            .build();
        assert_eq!(model.config.input_dim, 8);
        assert_eq!(model.config.hidden_dim, 16);
    }

    #[test]
    fn test_verify_equivariance() {
        let config = GeometricModelConfig {
            num_agents: 5,
            input_dim: 4,
            hidden_dim: 8,
            output_dim: 4,
            num_layers: 1,
            ..Default::default()
        };
        let model = EquivariantAgentModel::new(config);
        let graph = make_test_graph();
        let features = AgentFeatures::random(5, 4);
        let results = model.verify_equivariance(&graph, &features, 1e-6);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_full_pipeline() {
        let graph = AgentGraph::fully_connected(4);
        let features = AgentFeatures::random(4, 8);

        // Permutation layer
        let perm_layer = DeepSetsLayer::new(8, 6, Aggregation::Sum);
        let f1 = perm_layer.forward(&features);
        assert_eq!(f1.n_agents(), 4);

        // Message passing
        let mp_layer = MessagePassingLayer::new(6, 4, MessageAggregation::Sum);
        let f2 = mp_layer.forward(&graph, &f1);
        assert_eq!(f2.n_agents(), 4);

        // Scale normalization
        let scale = ScaleEquivLayer::new(4, 2, ScaleMode::Normalize);
        let f3 = scale.forward(&f2);
        assert_eq!(f3.n_agents(), 4);
    }
}
