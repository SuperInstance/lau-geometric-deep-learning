//! Spectral filters: convolution in frequency domain via graph Fourier transform.
//!
//! The graph Fourier transform diagonalizes the graph Laplacian L = UΛU^T,
//! enabling convolution in the frequency domain as U g(Λ) U^T x.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::core::AgentGraph;
use crate::core::random_vector;

/// Spectral filter that operates in the graph frequency domain.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpectralFilter {
    /// Number of filter coefficients (Chebyshev polynomial order).
    pub order: usize,
    /// Filter coefficients in the spectral domain.
    pub coefficients: DVector<f64>,
}

impl SpectralFilter {
    pub fn new(order: usize) -> Self {
        Self {
            order,
            coefficients: random_vector(order + 1),
        }
    }

    pub fn with_coefficients(coefficients: DVector<f64>) -> Self {
        let order = coefficients.nrows().saturating_sub(1);
        Self { order, coefficients }
    }

    /// Low-pass filter (smoothing).
    pub fn low_pass(order: usize, cutoff: f64) -> Self {
        let mut coeffs = DVector::zeros(order + 1);
        coeffs[0] = 1.0;
        for k in 1..=order {
            let lambda_k = k as f64 / order as f64;
            coeffs[k] = (-cutoff * lambda_k).exp();
        }
        Self { order, coefficients: coeffs }
    }

    /// High-pass filter (edge detection).
    pub fn high_pass(order: usize, cutoff: f64) -> Self {
        let mut coeffs = DVector::zeros(order + 1);
        coeffs[0] = 0.0;
        for k in 1..=order {
            let lambda_k = k as f64 / order as f64;
            coeffs[k] = 1.0 - (-cutoff * lambda_k).exp();
        }
        Self { order, coefficients: coeffs }
    }

    /// Apply filter via full eigendecomposition.
    pub fn apply_exact(&self, graph: &AgentGraph, features: &DMatrix<f64>) -> DMatrix<f64> {
        let lap = graph.normalized_laplacian();
        let n = lap.nrows();

        // Eigendecomposition (symmetric)
        let eig = lap.symmetric_eigen();
        let eigenvalues = eig.eigenvalues;
        let eigenvectors = eig.eigenvectors;

        // Apply filter in spectral domain
        let mut filtered_eigs = DMatrix::zeros(n, n);
        for i in 0..n {
            filtered_eigs[(i, i)] = self.evaluate(eigenvalues[i]);
        }

        // U * g(Λ) * U^T * X
        let ut_x = eigenvectors.transpose() * features;
        let g_ut_x = &filtered_eigs * &ut_x;
        eigenvectors * g_ut_x
    }

    /// Evaluate the spectral filter at a given eigenvalue.
    pub fn evaluate(&self, lambda: f64) -> f64 {
        // Polynomial filter: g(λ) = Σ_k c_k λ^k
        let mut result = 0.0;
        let mut lambda_k = 1.0;
        for k in 0..self.coefficients.nrows() {
            result += self.coefficients[k] * lambda_k;
            lambda_k *= lambda;
        }
        result
    }

    /// Apply filter via Chebyshev polynomial approximation (faster).
    pub fn apply_chebyshev(&self, graph: &AgentGraph, features: &DMatrix<f64>) -> DMatrix<f64> {
        let lap = graph.normalized_laplacian();
        let n = lap.nrows();
        let identity = DMatrix::identity(n, n);

        // Scale Laplacian to [-1, 1]: L_scaled = 2*L/λ_max - I
        let lambda_max = self.estimate_lambda_max(graph);
        let scaled_lap = &lap * (2.0 / lambda_max) - &identity;

        // Chebyshev polynomials: T_0 = I, T_1 = L_scaled, T_k = 2*L_scaled*T_{k-1} - T_{k-2}
        let mut t_prev = features.clone(); // T_0 * X = X
        let mut t_curr = &scaled_lap * features; // T_1 * X = L_scaled * X

        let mut result = &t_prev * self.coefficients[0] + &t_curr * self.coefficients[1];

        for k in 2..=self.order {
            if k >= self.coefficients.nrows() {
                break;
            }
            let t_next = &scaled_lap * &t_curr * 2.0 - &t_prev;
            result += &t_next * self.coefficients[k];
            t_prev = t_curr;
            t_curr = t_next;
        }

        result
    }

    /// Estimate largest eigenvalue using power iteration.
    fn estimate_lambda_max(&self, graph: &AgentGraph) -> f64 {
        let lap = graph.normalized_laplacian();
        let n = lap.nrows();
        if n == 0 {
            return 2.0;
        }

        let mut v = random_vector(n);
        let mut lambda = 0.0;

        for _ in 0..50 {
            v.normalize_mut();
            let w = &lap * &v;
            lambda = v.dot(&w);
            v = w;
        }

        lambda.max(2.0) // Clamp to at least 2 for normalized Laplacian
    }
}

/// Graph Fourier Transform.
pub struct GraphFourier;

impl GraphFourier {
    /// Compute the graph Fourier transform of a signal.
    /// GFT(f) = U^T f where L = UΛU^T.
    pub fn transform(graph: &AgentGraph, signal: &DVector<f64>) -> DVector<f64> {
        let lap = graph.normalized_laplacian();
        let eig = lap.symmetric_eigen();
        eig.eigenvectors.transpose() * signal
    }

    /// Inverse graph Fourier transform.
    pub fn inverse_transform(graph: &AgentGraph, spectrum: &DVector<f64>) -> DVector<f64> {
        let lap = graph.normalized_laplacian();
        let eig = lap.symmetric_eigen();
        eig.eigenvectors * spectrum
    }

    /// Compute graph frequencies (eigenvalues of the Laplacian).
    pub fn frequencies(graph: &AgentGraph) -> DVector<f64> {
        let lap = graph.normalized_laplacian();
        let eig = lap.symmetric_eigen();
        eig.eigenvalues
    }

    /// Spectral pooling: keep only low-frequency components.
    pub fn spectral_pool(graph: &AgentGraph, signal: &DVector<f64>, keep_ratio: f64) -> DVector<f64> {
        let spectrum = Self::transform(graph, signal);
        let n = spectrum.nrows();
        let keep = ((n as f64) * keep_ratio).ceil() as usize;

        let mut filtered = spectrum.clone();
        for i in keep..n {
            filtered[i] = 0.0;
        }
        Self::inverse_transform(graph, &filtered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_filter_new() {
        let f = SpectralFilter::new(3);
        assert_eq!(f.order, 3);
        assert_eq!(f.coefficients.nrows(), 4);
    }

    #[test]
    fn test_spectral_filter_evaluate() {
        let f = SpectralFilter::with_coefficients(DVector::from_vec(vec![1.0, 0.0]));
        assert!((f.evaluate(0.5) - 1.0).abs() < 1e-10);
        assert!((f.evaluate(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_filter_polynomial() {
        let f = SpectralFilter::with_coefficients(DVector::from_vec(vec![0.0, 1.0, 0.0]));
        // g(λ) = λ
        assert!((f.evaluate(2.0) - 2.0).abs() < 1e-10);
        assert!((f.evaluate(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_filter_exact() {
        let graph = AgentGraph::ring(5);
        let filter = SpectralFilter::with_coefficients(DVector::from_vec(vec![1.0, 0.5]));
        let features = DMatrix::new_random(5, 3);
        let result = filter.apply_exact(&graph, &features);
        assert_eq!(result.nrows(), 5);
        assert_eq!(result.ncols(), 3);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_spectral_filter_chebyshev() {
        let graph = AgentGraph::ring(5);
        let filter = SpectralFilter::new(3);
        let features = DMatrix::new_random(5, 2);
        let result = filter.apply_chebyshev(&graph, &features);
        assert_eq!(result.nrows(), 5);
        assert_eq!(result.ncols(), 2);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_spectral_low_pass() {
        let f = SpectralFilter::low_pass(5, 2.0);
        // The polynomial coefficients should decay (exponential envelope)
        assert!(f.coefficients[0] > 0.0);
        assert!(f.coefficients[1] > 0.0);
    }

    #[test]
    fn test_spectral_high_pass() {
        let f = SpectralFilter::high_pass(5, 2.0);
        let low = f.evaluate(0.1);
        let high = f.evaluate(1.0);
        assert!(high > low);
    }

    #[test]
    fn test_graph_fourier_transform() {
        let graph = AgentGraph::ring(6);
        let signal = DVector::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let spectrum = GraphFourier::transform(&graph, &signal);
        assert_eq!(spectrum.nrows(), 6);
    }

    #[test]
    fn test_graph_fourier_roundtrip() {
        let graph = AgentGraph::ring(6);
        let signal = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let spectrum = GraphFourier::transform(&graph, &signal);
        let recovered = GraphFourier::inverse_transform(&graph, &spectrum);
        for i in 0..6 {
            assert!((recovered[i] - signal[i]).abs() < 1e-8);
        }
    }

    #[test]
    fn test_graph_frequencies() {
        let graph = AgentGraph::ring(5);
        let freqs = GraphFourier::frequencies(&graph);
        assert_eq!(freqs.nrows(), 5);
        // Eigenvalues should be non-negative (Laplacian is PSD)
        assert!(freqs.iter().all(|&v| v >= -1e-8));
    }

    #[test]
    fn test_spectral_pool() {
        let graph = AgentGraph::ring(6);
        let signal = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let pooled = GraphFourier::spectral_pool(&graph, &signal, 0.5);
        assert_eq!(pooled.nrows(), 6);
        // Should still have a valid signal
        assert!(pooled.iter().any(|v| v.abs() > 0.1));
    }
}
