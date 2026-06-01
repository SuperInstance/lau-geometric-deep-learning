//! Time equivariance: features that respect temporal symmetries.
//!
//! For agent systems evolving in time, we want features that are
//! equivariant to time shifts (T_t ∘ f = f ∘ T_t).

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::core::GroupAction;
use crate::core::random_vector;

/// A time shift.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeShift {
    /// Number of steps to shift (can be negative).
    pub steps: i64,
}

impl TimeShift {
    pub fn new(steps: i64) -> Self {
        Self { steps }
    }
}

impl GroupAction for TimeShift {
    fn act(&self, features: &DMatrix<f64>) -> DMatrix<f64> {
        let n = features.nrows();
        let d = features.ncols();
        let mut result = DMatrix::zeros(n, d);
        for i in 0..n {
            let src = ((i as i64 + self.steps).rem_euclid(n as i64)) as usize;
            for j in 0..d {
                result[(i, j)] = features[(src, j)];
            }
        }
        result
    }

    fn inverse(&self) -> Self {
        Self { steps: -self.steps }
    }

    fn compose(&self, other: &Self) -> Self {
        Self { steps: self.steps + other.steps }
    }

    fn identity() -> Self {
        Self { steps: 0 }
    }
}

/// Temporal convolution layer (shift-equivariant).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalConvLayer {
    pub kernel_size: usize,
    pub kernel: DVector<f64>,
    pub input_channels: usize,
    pub output_channels: usize,
    pub weights: DMatrix<f64>,
}

impl TemporalConvLayer {
    pub fn new(input_channels: usize, output_channels: usize, kernel_size: usize) -> Self {
        Self {
            kernel_size,
            kernel: random_vector(kernel_size),
            input_channels,
            output_channels,
            weights: DMatrix::new_random(output_channels, input_channels),
        }
    }

    /// 1D temporal convolution (causal).
    pub fn convolve_causal(&self, signal: &DVector<f64>) -> DVector<f64> {
        let n = signal.nrows();
        let k = self.kernel_size;
        let mut result = DVector::zeros(n);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..k {
                if i >= j {
                    sum += signal[i - j] * self.kernel[j];
                }
            }
            result[i] = sum;
        }
        result
    }

    /// Apply temporal convolution to agent time series.
    /// Input: (seq_len × features), Output: (seq_len × output_channels)
    pub fn forward(&self, time_series: &DMatrix<f64>) -> DMatrix<f64> {
        let seq_len = time_series.nrows();
        let feat_dim = time_series.ncols();
        let mut result = DMatrix::zeros(seq_len, self.output_channels);

        // Apply kernel to each feature dimension, then project
        for j in 0..feat_dim.min(self.input_channels) {
            let col: DVector<f64> = time_series.column(j).clone_owned();
            let convolved = self.convolve_causal(&col);
            for i in 0..seq_len {
                for oc in 0..self.output_channels {
                    result[(i, oc)] += convolved[i] * self.weights[(oc, j % self.weights.ncols())];
                }
            }
        }
        result
    }
}

/// Recurrent equivariant layer that respects temporal symmetry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalRecurrentLayer {
    pub hidden_dim: usize,
    pub input_dim: usize,
    pub w_ih: DMatrix<f64>,
    pub w_hh: DMatrix<f64>,
}

impl TemporalRecurrentLayer {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            hidden_dim,
            input_dim,
            w_ih: DMatrix::new_random(hidden_dim, input_dim),
            w_hh: DMatrix::new_random(hidden_dim, hidden_dim),
        }
    }

    /// Process a sequence, returning hidden states.
    pub fn forward(&self, sequence: &DMatrix<f64>) -> DMatrix<f64> {
        let seq_len = sequence.nrows();
        let mut h = DVector::zeros(self.hidden_dim);
        let mut outputs = DMatrix::zeros(seq_len, self.hidden_dim);

        for t in 0..seq_len {
            let x_t: DVector<f64> = sequence.row(t).transpose();
            let pre = &self.w_ih * &x_t + &self.w_hh * &h;
            // ReLU
            h = pre.map(|v| if v > 0.0 { v } else { 0.0 });
            for j in 0..self.hidden_dim {
                outputs[(t, j)] = h[j];
            }
        }
        outputs
    }

    /// Process in reverse (for bidirectional).
    pub fn forward_reverse(&self, sequence: &DMatrix<f64>) -> DMatrix<f64> {
        let seq_len = sequence.nrows();
        let mut h = DVector::zeros(self.hidden_dim);
        let mut outputs = DMatrix::zeros(seq_len, self.hidden_dim);

        for t in (0..seq_len).rev() {
            let x_t: DVector<f64> = sequence.row(t).transpose();
            let pre = &self.w_ih * &x_t + &self.w_hh * &h;
            h = pre.map(|v| if v > 0.0 { v } else { 0.0 });
            for j in 0..self.hidden_dim {
                outputs[(t, j)] = h[j];
            }
        }
        outputs
    }
}

/// Compute temporal differences (equivariant to time shifts).
pub fn temporal_differences(time_series: &DMatrix<f64>, order: usize) -> DMatrix<f64> {
    let n = time_series.nrows();
    let d = time_series.ncols();
    let out_n = n.saturating_sub(order);
    let mut result = DMatrix::zeros(out_n, d);

    // Start with original
    let mut current = time_series.clone();
    for _ in 0..order {
        let cn = current.nrows().saturating_sub(1);
        let mut diff = DMatrix::zeros(cn, d);
        for i in 0..cn {
            for j in 0..d {
                diff[(i, j)] = current[(i + 1, j)] - current[(i, j)];
            }
        }
        current = diff;
    }

    result.copy_from(&current);
    result
}

/// Compute running average (time-equivariant smoothing).
pub fn running_average(time_series: &DMatrix<f64>, window: usize) -> DMatrix<f64> {
    let n = time_series.nrows();
    let d = time_series.ncols();
    let mut result = DMatrix::zeros(n, d);
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let count = i - start + 1;
        for j in 0..d {
            let sum: f64 = (start..=i).map(|t| time_series[(t, j)]).sum();
            result[(i, j)] = sum / count as f64;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_shift_forward() {
        let ts = TimeShift::new(1);
        let features = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let shifted = ts.act(&features);
        // Shift by 1: row i gets from row (i+1)%3
        assert!((shifted[(0, 0)] - 3.0).abs() < 1e-10); // row 0 ← row 1
        assert!((shifted[(1, 0)] - 5.0).abs() < 1e-10); // row 1 ← row 2
        assert!((shifted[(2, 0)] - 1.0).abs() < 1e-10); // row 2 ← row 0
    }

    #[test]
    fn test_time_shift_inverse() {
        let ts = TimeShift::new(3);
        let inv = ts.inverse();
        assert_eq!(inv.steps, -3);
    }

    #[test]
    fn test_time_shift_compose() {
        let t1 = TimeShift::new(2);
        let t2 = TimeShift::new(3);
        let comp = t1.compose(&t2);
        assert_eq!(comp.steps, 5);
    }

    #[test]
    fn test_time_shift_identity() {
        let ts = TimeShift::identity();
        assert_eq!(ts.steps, 0);
    }

    #[test]
    fn test_temporal_conv_causal() {
        let mut layer = TemporalConvLayer::new(2, 2, 3);
        layer.kernel = DVector::from_vec(vec![1.0, 0.0, 0.0]); // Identity
        let signal = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = layer.convolve_causal(&signal);
        for i in 0..4 {
            assert!((result[i] - signal[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_temporal_conv_forward() {
        let layer = TemporalConvLayer::new(2, 3, 3);
        let ts = DMatrix::new_random(5, 2);
        let result = layer.forward(&ts);
        assert_eq!(result.nrows(), 5);
        assert_eq!(result.ncols(), 3);
    }

    #[test]
    fn test_temporal_recurrent_forward() {
        let layer = TemporalRecurrentLayer::new(3, 4);
        let seq = DMatrix::new_random(6, 3);
        let output = layer.forward(&seq);
        assert_eq!(output.nrows(), 6);
        assert_eq!(output.ncols(), 4);
    }

    #[test]
    fn test_temporal_recurrent_reverse() {
        let layer = TemporalRecurrentLayer::new(3, 4);
        let seq = DMatrix::new_random(6, 3);
        let output = layer.forward_reverse(&seq);
        assert_eq!(output.nrows(), 6);
        assert_eq!(output.ncols(), 4);
    }

    #[test]
    fn test_temporal_differences_order1() {
        let ts = DMatrix::from_row_slice(4, 1, &[1.0, 3.0, 6.0, 10.0]);
        let diff = temporal_differences(&ts, 1);
        assert_eq!(diff.nrows(), 3);
        assert!((diff[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((diff[(1, 0)] - 3.0).abs() < 1e-10);
        assert!((diff[(2, 0)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_differences_order2() {
        let ts = DMatrix::from_row_slice(5, 1, &[1.0, 3.0, 6.0, 10.0, 15.0]);
        let diff = temporal_differences(&ts, 2);
        assert_eq!(diff.nrows(), 3);
        // First diff: [2,3,4,5], second: [1,1,1]
        assert!((diff[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((diff[(1, 0)] - 1.0).abs() < 1e-10);
        assert!((diff[(2, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_running_average() {
        let ts = DMatrix::from_row_slice(4, 1, &[1.0, 2.0, 3.0, 4.0]);
        let avg = running_average(&ts, 2);
        assert_eq!(avg.nrows(), 4);
        assert!((avg[(0, 0)] - 1.0).abs() < 1e-10); // just [1]
        assert!((avg[(1, 0)] - 1.5).abs() < 1e-10); // [1,2]
        assert!((avg[(2, 0)] - 2.5).abs() < 1e-10); // [2,3]
        assert!((avg[(3, 0)] - 3.5).abs() < 1e-10); // [3,4]
    }
}
