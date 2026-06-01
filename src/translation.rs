//! Translation equivariance: shift-invariant agent features.
//!
//! A function f is translation equivariant if f(T_s·x) = T_s·f(x)
//! for any translation T_s by shift vector s.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::core::{AgentFeatures, EquivariantLayer, GroupAction};
use crate::core::random_vector;

/// A translation in feature space.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Translation {
    pub shift: DVector<f64>,
}

impl Translation {
    pub fn new(shift: DVector<f64>) -> Self {
        Self { shift }
    }

    pub fn zero(dim: usize) -> Self {
        Self { shift: DVector::zeros(dim) }
    }

    pub fn dim(&self) -> usize {
        self.shift.nrows()
    }
}

impl GroupAction for Translation {
    fn act(&self, features: &DMatrix<f64>) -> DMatrix<f64> {
        let mut result = features.clone();
        for mut row in result.row_iter_mut() {
            for (j, s) in self.shift.iter().enumerate() {
                if j < row.ncols() {
                    row[j] += s;
                }
            }
        }
        result
    }

    fn inverse(&self) -> Self {
        Self { shift: -&self.shift }
    }

    fn compose(&self, other: &Self) -> Self {
        Self { shift: &self.shift + &other.shift }
    }

    fn identity() -> Self {
        Self::zero(0)
    }
}

/// Translation equivariant linear layer.
/// Uses weight sharing: only relative differences matter.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TranslationEquivLayer {
    /// Weight matrix applied to relative features.
    pub weights: DMatrix<f64>,
    /// Bias (must be zero for strict equivariance).
    pub use_bias: bool,
    pub bias: DVector<f64>,
}

impl TranslationEquivLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weights: DMatrix::new_random(output_dim, input_dim),
            use_bias: false,
            bias: DVector::zeros(output_dim),
        }
    }

    pub fn with_bias(mut self) -> Self {
        self.use_bias = true;
        self.bias = random_vector(self.weights.nrows());
        self
    }

    /// Forward pass using relative coordinates.
    /// For each agent i, computes f(x_i - x_mean) where x_mean is the centroid.
    pub fn forward_relative(&self, features: &AgentFeatures) -> AgentFeatures {
        let n = features.n_agents();
        let d = features.feature_dim();

        // Compute centroid
        let mut centroid = DVector::zeros(d);
        for i in 0..n {
            for j in 0..d {
                centroid[j] += features.data[(i, j)];
            }
        }
        centroid = &centroid / n as f64;

        // Compute relative features
        let mut result = DMatrix::zeros(n, self.weights.nrows());
        for i in 0..n {
            let mut rel = DVector::zeros(d);
            for j in 0..d {
                rel[j] = features.data[(i, j)] - centroid[j];
            }
            let out = &self.weights * &rel;
            for j in 0..out.nrows().min(result.ncols()) {
                result[(i, j)] = out[j];
            }
        }
        AgentFeatures::from_matrix(result)
    }

    /// Forward pass using differences between agents.
    pub fn forward_differences(&self, features: &AgentFeatures) -> AgentFeatures {
        let n = features.n_agents();
        let d = features.feature_dim();
        let out_dim = self.weights.nrows();

        let mut result = DMatrix::zeros(n, out_dim);
        for i in 0..n {
            // Sum of differences with all other agents
            let mut diff_sum = DVector::zeros(d);
            for j in 0..n {
                if i != j {
                    for k in 0..d {
                        diff_sum[k] += features.data[(i, k)] - features.data[(j, k)];
                    }
                }
            }
            if n > 1 {
                diff_sum = &diff_sum / (n - 1) as f64;
            }
            let out = &self.weights * &diff_sum;
            for k in 0..out.nrows().min(out_dim) {
                result[(i, k)] = out[k];
            }
        }
        AgentFeatures::from_matrix(result)
    }
}

impl EquivariantLayer for TranslationEquivLayer {
    type Group = Translation;

    fn forward(&self, features: &DMatrix<f64>) -> DMatrix<f64> {
        let af = AgentFeatures::from_matrix(features.clone());
        self.forward_relative(&af).data
    }
}

/// Circular convolution for translation equivariance on ring graphs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircularConvLayer {
    pub kernel_size: usize,
    pub kernel: DVector<f64>,
    pub input_dim: usize,
}

impl CircularConvLayer {
    pub fn new(input_dim: usize, kernel_size: usize) -> Self {
        Self {
            kernel_size,
            kernel: random_vector(kernel_size),
            input_dim,
        }
    }

    /// 1D circular convolution.
    pub fn convolve1d(&self, signal: &DVector<f64>) -> DVector<f64> {
        let n = signal.nrows();
        let k = self.kernel_size;
        let mut result = DVector::zeros(n);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..k {
                let idx = (i + n - j) % n;
                sum += signal[idx] * self.kernel[j];
            }
            result[i] = sum;
        }
        result
    }

    /// Apply to each feature dimension of agent features.
    pub fn forward(&self, features: &AgentFeatures) -> AgentFeatures {
        let n = features.n_agents();
        let d = features.feature_dim();
        let mut result = DMatrix::zeros(n, d);
        for j in 0..d {
            let col: DVector<f64> = features.data.column(j).clone_owned();
            let convolved = self.convolve1d(&col);
            for i in 0..n {
                result[(i, j)] = convolved[i];
            }
        }
        AgentFeatures::from_matrix(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translation_act() {
        let t = Translation::new(DVector::from_vec(vec![1.0, 2.0]));
        let features = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
        let result = t.act(&features);
        assert!((result[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((result[(0, 1)] - 2.0).abs() < 1e-10);
        assert!((result[(1, 0)] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_translation_inverse() {
        let t = Translation::new(DVector::from_vec(vec![1.0, 2.0]));
        let inv = t.inverse();
        assert!((inv.shift[0] - (-1.0)).abs() < 1e-10);
        assert!((inv.shift[1] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_translation_compose() {
        let t1 = Translation::new(DVector::from_vec(vec![1.0]));
        let t2 = Translation::new(DVector::from_vec(vec![2.0]));
        let comp = t1.compose(&t2);
        assert!((comp.shift[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_translation_equiv_layer_forward() {
        let layer = TranslationEquivLayer::new(3, 4);
        let features = AgentFeatures::from_matrix(DMatrix::new_random(5, 3));
        let result = layer.forward_relative(&features);
        assert_eq!(result.n_agents(), 5);
        assert_eq!(result.feature_dim(), 4);
    }

    #[test]
    fn test_translation_equiv_relative() {
        // Adding same constant to all agents should not change relative features
        let layer = TranslationEquivLayer::new(2, 3);
        let f1 = AgentFeatures::from_matrix(DMatrix::from_row_slice(3, 2, &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]));
        let f2 = AgentFeatures::from_matrix(DMatrix::from_row_slice(3, 2, &[
            11.0, 12.0,
            13.0, 14.0,
            15.0, 16.0,
        ]));
        let r1 = layer.forward_relative(&f1);
        let r2 = layer.forward_relative(&f2);
        // Results should be identical since relative positions are the same
        for i in 0..3 {
            for j in 0..3 {
                assert!((r1.data[(i, j)] - r2.data[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_translation_equiv_differences() {
        let layer = TranslationEquivLayer::new(2, 3);
        let f1 = AgentFeatures::from_matrix(DMatrix::from_row_slice(3, 2, &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]));
        let f2 = AgentFeatures::from_matrix(DMatrix::from_row_slice(3, 2, &[
            11.0, 12.0,
            13.0, 14.0,
            15.0, 16.0,
        ]));
        let r1 = layer.forward_differences(&f1);
        let r2 = layer.forward_differences(&f2);
        // Same relative differences → same output
        for i in 0..3 {
            for j in 0..r1.feature_dim() {
                assert!((r1.data[(i, j)] - r2.data[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_circular_conv_identity_kernel() {
        let mut layer = CircularConvLayer::new(3, 3);
        // Dirac kernel: [1, 0, 0]
        layer.kernel = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let signal = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = layer.convolve1d(&signal);
        for i in 0..5 {
            assert!((result[i] - signal[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_circular_conv_shift() {
        let mut layer = CircularConvLayer::new(3, 3);
        // Shift kernel: [0, 0, 1] shifts signal by 1
        layer.kernel = DVector::from_vec(vec![0.0, 0.0, 1.0]);
        let signal = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = layer.convolve1d(&signal);
        // signal[i] * kernel[0] + signal[(i-1)%n] * kernel[1] + signal[(i-2)%n] * kernel[2]
        // = 0 + 0 + signal[(i-2)%4]
        assert!((result[0] - signal[2]).abs() < 1e-10);
        assert!((result[1] - signal[3]).abs() < 1e-10);
        assert!((result[2] - signal[0]).abs() < 1e-10);
    }

    #[test]
    fn test_circular_conv_forward() {
        let layer = CircularConvLayer::new(2, 3);
        let features = AgentFeatures::from_matrix(DMatrix::new_random(5, 2));
        let result = layer.forward(&features);
        assert_eq!(result.n_agents(), 5);
        assert_eq!(result.feature_dim(), 2);
    }

    #[test]
    fn test_translation_check_equivariance() {
        let layer = TranslationEquivLayer::new(3, 4);
        let features = AgentFeatures::from_matrix(DMatrix::new_random(5, 3));
        let shift = Translation::new(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        let r1 = layer.forward_relative(&features);
        let shifted = AgentFeatures::from_matrix(shift.act(&features.data));
        let r2 = layer.forward_relative(&shifted);
        // Should be identical
        for i in 0..r1.n_agents() {
            for j in 0..r1.feature_dim() {
                assert!((r1.data[(i, j)] - r2.data[(i, j)]).abs() < 1e-10);
            }
        }
    }
}
