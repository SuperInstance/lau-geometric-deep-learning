//! Scale equivariance: multi-resolution agent features.
//!
//! Features that behave predictably under scaling of the agent configuration.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::core::{AgentFeatures, EquivariantLayer, GroupAction};

/// A scaling transformation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScaleAction {
    pub factor: f64,
}

impl ScaleAction {
    pub fn new(factor: f64) -> Self {
        Self { factor }
    }
}

impl GroupAction for ScaleAction {
    fn act(&self, features: &DMatrix<f64>) -> DMatrix<f64> {
        features * self.factor
    }

    fn inverse(&self) -> Self {
        Self { factor: 1.0 / self.factor }
    }

    fn compose(&self, other: &Self) -> Self {
        Self { factor: self.factor * other.factor }
    }

    fn identity() -> Self {
        Self { factor: 1.0 }
    }
}

/// Scale-equivariant layer.
/// Uses normalization to achieve scale invariance/equivariance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScaleEquivLayer {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: DMatrix<f64>,
    pub mode: ScaleMode,
}

/// How to handle scale.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ScaleMode {
    /// Normalize by L2 norm of input.
    Normalize,
    /// Use log-scale features.
    LogScale,
    /// Multi-scale: process at multiple scales and combine.
    MultiScale { scales: Vec<f64> },
}

impl ScaleEquivLayer {
    pub fn new(input_dim: usize, output_dim: usize, mode: ScaleMode) -> Self {
        Self {
            input_dim,
            output_dim,
            weights: DMatrix::new_random(output_dim, input_dim),
            mode,
        }
    }

    /// Forward pass.
    pub fn forward(&self, features: &AgentFeatures) -> AgentFeatures {
        match &self.mode {
            ScaleMode::Normalize => self.forward_normalize(features),
            ScaleMode::LogScale => self.forward_log(features),
            ScaleMode::MultiScale { scales } => self.forward_multiscale(features, scales),
        }
    }

    fn forward_normalize(&self, features: &AgentFeatures) -> AgentFeatures {
        let norm = features.norm();
        let scale = if norm > 1e-10 { 1.0 / norm } else { 1.0 };
        let normalized = features.scale(scale);
        let n = normalized.n_agents();
        let mut result = DMatrix::zeros(n, self.output_dim);
        for i in 0..n {
            let row: DVector<f64> = normalized.data.row(i).transpose();
            let out = &self.weights * &row;
            for j in 0..self.output_dim.min(out.nrows()) {
                result[(i, j)] = out[j];
            }
        }
        AgentFeatures::from_matrix(result)
    }

    fn forward_log(&self, features: &AgentFeatures) -> AgentFeatures {
        let n = features.n_agents();
        let d = features.feature_dim();
        let mut log_data = DMatrix::zeros(n, d);
        for i in 0..n {
            for j in 0..d {
                log_data[(i, j)] = features.data[(i, j)].abs().max(1e-10).ln();
            }
        }
        let mut result = DMatrix::zeros(n, self.output_dim);
        for i in 0..n {
            let row: DVector<f64> = log_data.row(i).transpose();
            let out = &self.weights * &row;
            for j in 0..self.output_dim.min(out.nrows()) {
                result[(i, j)] = out[j];
            }
        }
        AgentFeatures::from_matrix(result)
    }

    fn forward_multiscale(&self, features: &AgentFeatures, scales: &[f64]) -> AgentFeatures {
        let dim_per_scale = self.output_dim / scales.len().max(1);
        let mut combined = DMatrix::zeros(features.n_agents(), self.output_dim);

        for (si, &s) in scales.iter().enumerate() {
            let scaled = features.scale(s);
            let n = scaled.n_agents();
            let d = scaled.feature_dim();
            let offset = si * dim_per_scale;

            for i in 0..n {
                let row: DVector<f64> = scaled.data.row(i).transpose();
                let weight_cols = self.weights.columns(0, d.min(self.weights.ncols()));
                let out = &weight_cols * &row;
                for j in 0..dim_per_scale.min(out.nrows()) {
                    if offset + j < self.output_dim {
                        combined[(i, offset + j)] = out[j];
                    }
                }
            }
        }

        AgentFeatures::from_matrix(combined)
    }
}

impl EquivariantLayer for ScaleEquivLayer {
    type Group = ScaleAction;

    fn forward(&self, features: &DMatrix<f64>) -> DMatrix<f64> {
        let af = AgentFeatures::from_matrix(features.clone());
        self.forward(&af).data
    }
}

/// Multi-resolution pooling: aggregate features at different scales.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiResolutionPool {
    pub scales: Vec<f64>,
}

impl MultiResolutionPool {
    pub fn new(scales: Vec<f64>) -> Self {
        Self { scales }
    }

    /// Pool features at each scale.
    pub fn pool(&self, _positions: &DMatrix<f64>, features: &AgentFeatures) -> Vec<AgentFeatures> {
        self.scales.iter().map(|&s| {
            let n = features.n_agents();
            let mut pooled = features.data.clone();
            for i in 0..n {
                for j in 0..features.feature_dim() {
                    pooled[(i, j)] *= s.sqrt().max(0.1);
                }
            }
            AgentFeatures::from_matrix(pooled)
        }).collect()
    }

    /// Concatenate multi-scale features.
    pub fn pool_concat(&self, positions: &DMatrix<f64>, features: &AgentFeatures) -> AgentFeatures {
        let pooled = self.pool(positions, features);
        if pooled.is_empty() {
            return features.clone();
        }
        let n = features.n_agents();
        let d = features.feature_dim();
        let total_dim = d * pooled.len();
        let mut result = DMatrix::zeros(n, total_dim);
        for (si, pf) in pooled.iter().enumerate() {
            for i in 0..n {
                for j in 0..d {
                    result[(i, si * d + j)] = pf.data[(i, j)];
                }
            }
        }
        AgentFeatures::from_matrix(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_action() {
        let s = ScaleAction::new(2.0);
        let features = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let result = s.act(&features);
        assert!((result[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((result[(1, 1)] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_scale_inverse() {
        let s = ScaleAction::new(3.0);
        let inv = s.inverse();
        assert!((inv.factor - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_scale_compose() {
        let s1 = ScaleAction::new(2.0);
        let s2 = ScaleAction::new(3.0);
        let comp = s1.compose(&s2);
        assert!((comp.factor - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_scale_identity() {
        let s = ScaleAction::identity();
        assert!((s.factor - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_scale_normalize_invariance() {
        let layer = ScaleEquivLayer::new(3, 4, ScaleMode::Normalize);
        let f1 = AgentFeatures::from_matrix(DMatrix::from_row_slice(3, 3, &[
            1.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 3.0,
        ]));
        let f2 = f1.scale(5.0);
        let r1 = layer.forward(&f1);
        let r2 = layer.forward(&f2);
        for i in 0..3 {
            for j in 0..4 {
                assert!((r1.data[(i, j)] - r2.data[(i, j)]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn test_scale_log_mode() {
        let layer = ScaleEquivLayer::new(2, 3, ScaleMode::LogScale);
        let features = AgentFeatures::from_matrix(DMatrix::from_row_slice(2, 2, &[
            1.0, 2.0,
            3.0, 4.0,
        ]));
        let result = layer.forward(&features);
        assert_eq!(result.n_agents(), 2);
        assert_eq!(result.feature_dim(), 3);
    }

    #[test]
    fn test_scale_multiscale() {
        let layer = ScaleEquivLayer::new(2, 6, ScaleMode::MultiScale {
            scales: vec![0.5, 1.0, 2.0],
        });
        let features = AgentFeatures::from_matrix(DMatrix::new_random(4, 2));
        let result = layer.forward(&features);
        assert_eq!(result.n_agents(), 4);
        assert_eq!(result.feature_dim(), 6);
    }

    #[test]
    fn test_multi_resolution_pool() {
        let pool = MultiResolutionPool::new(vec![0.5, 1.0, 2.0]);
        let positions = DMatrix::new_random(4, 2);
        let features = AgentFeatures::from_matrix(DMatrix::new_random(4, 3));
        let pooled = pool.pool(&positions, &features);
        assert_eq!(pooled.len(), 3);
        for p in &pooled {
            assert_eq!(p.n_agents(), 4);
        }
    }

    #[test]
    fn test_multi_resolution_concat() {
        let pool = MultiResolutionPool::new(vec![1.0, 2.0]);
        let positions = DMatrix::new_random(3, 2);
        let features = AgentFeatures::from_matrix(DMatrix::new_random(3, 4));
        let result = pool.pool_concat(&positions, &features);
        assert_eq!(result.n_agents(), 3);
        assert_eq!(result.feature_dim(), 8);
    }

    #[test]
    fn test_scale_equivariance_normalize() {
        let layer = ScaleEquivLayer::new(2, 3, ScaleMode::Normalize);
        let f = AgentFeatures::from_matrix(DMatrix::from_row_slice(2, 2, &[
            1.0, 2.0,
            3.0, 4.0,
        ]));
        let r1 = layer.forward(&f);
        let scaled = f.scale(10.0);
        let r2 = layer.forward(&scaled);
        for i in 0..2 {
            for j in 0..3 {
                assert!((r1.data[(i, j)] - r2.data[(i, j)]).abs() < 1e-8);
            }
        }
    }
}
