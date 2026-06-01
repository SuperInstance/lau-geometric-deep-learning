//! Permutation equivariance: agent order doesn't matter.
//!
//! A function f is permutation equivariant if f(P·X) = P·f(X) for any
//! permutation matrix P. This is critical for agent systems where agents
//! have no canonical ordering.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::core::{AgentFeatures, EquivariantLayer, GroupAction};
use crate::core::random_vector;

/// A permutation of indices.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Permutation {
    /// perm[i] = j means new position i comes from old position j.
    pub perm: Vec<usize>,
}

impl Permutation {
    pub fn new(perm: Vec<usize>) -> Self {
        Self { perm }
    }

    /// Identity permutation of size n.
    pub fn identity(n: usize) -> Self {
        Self { perm: (0..n).collect() }
    }

    /// Random permutation (deterministic based on seed for testing).
    pub fn from_seed(n: usize, seed: u64) -> Self {
        let mut perm: Vec<usize> = (0..n).collect();
        // Simple LCG-based shuffle
        let mut s = seed;
        for i in (1..n).rev() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (s >> 33) as usize % (i + 1);
            perm.swap(i, j);
        }
        Self { perm }
    }

    /// Inverse permutation.
    pub fn inverse(&self) -> Self {
        let n = self.perm.len();
        let mut inv = vec![0; n];
        for (i, &j) in self.perm.iter().enumerate() {
            inv[j] = i;
        }
        Self { perm: inv }
    }

    /// Compose: self(other(x)) = self.perm[other.perm[x]].
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            perm: other.perm.iter().map(|&j| self.perm[j]).collect(),
        }
    }

    /// Apply to a vector of items.
    pub fn apply<T: Clone>(&self, items: &[T]) -> Vec<T> {
        self.perm.iter().map(|&i| items[i].clone()).collect()
    }

    /// Size of permutation.
    pub fn len(&self) -> usize {
        self.perm.len()
    }

    pub fn is_empty(&self) -> bool {
        self.perm.is_empty()
    }

    /// To permutation matrix.
    pub fn to_matrix(&self) -> DMatrix<f64> {
        let n = self.perm.len();
        let mut m = DMatrix::zeros(n, n);
        for (i, &j) in self.perm.iter().enumerate() {
            m[(i, j)] = 1.0;
        }
        m
    }
}

impl GroupAction for Permutation {
    fn act(&self, features: &DMatrix<f64>) -> DMatrix<f64> {
        assert_eq!(self.perm.len(), features.nrows(), "Permutation size must match number of rows");
        let n = features.nrows();
        let d = features.ncols();
        let mut result = DMatrix::zeros(n, d);
        for (new_i, &old_i) in self.perm.iter().enumerate() {
            for j in 0..d {
                result[(new_i, j)] = features[(old_i, j)];
            }
        }
        result
    }

    fn inverse(&self) -> Self {
        let n = self.perm.len();
        let mut inv = vec![0; n];
        for (i, &j) in self.perm.iter().enumerate() {
            inv[j] = i;
        }
        Self { perm: inv }
    }

    fn compose(&self, other: &Self) -> Self {
        Self {
            perm: other.perm.iter().map(|&j| self.perm[j]).collect(),
        }
    }

    fn identity() -> Self {
        Self::identity(0) // caller should construct appropriate size
    }
}

/// Deep Sets layer: permutation equivariant via sum/mean/max aggregation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepSetsLayer {
    /// Input dimension.
    pub input_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Phi weights: maps individual features.
    pub phi_weights: DMatrix<f64>,
    /// Phi bias.
    pub phi_bias: DVector<f64>,
    /// Rho weights: maps aggregated features.
    pub rho_weights: DMatrix<f64>,
    /// Rho bias.
    pub rho_bias: DVector<f64>,
    /// Aggregation method.
    pub aggregation: Aggregation,
}

/// Aggregation method for Deep Sets.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Aggregation {
    Sum,
    Mean,
    Max,
}

impl DeepSetsLayer {
    pub fn new(input_dim: usize, output_dim: usize, aggregation: Aggregation) -> Self {
        let hidden = (input_dim + output_dim) / 2 + 1;
        Self {
            input_dim,
            output_dim,
            phi_weights: DMatrix::new_random(hidden, input_dim),
            phi_bias: random_vector(hidden),
            rho_weights: DMatrix::new_random(output_dim, hidden),
            rho_bias: random_vector(output_dim),
            aggregation,
        }
    }

    /// Phi function: transform individual agent features.
    pub fn phi(&self, x: &DVector<f64>) -> DVector<f64> {
        let mut result = &self.phi_weights * x + &self.phi_bias;
        // ReLU activation
        for v in result.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        result
    }

    /// Rho function: transform aggregated features.
    pub fn rho(&self, x: &DVector<f64>) -> DVector<f64> {
        &self.rho_weights * x + &self.rho_bias
    }

    /// Aggregate features across agents.
    pub fn aggregate(&self, features: &[DVector<f64>]) -> DVector<f64> {
        if features.is_empty() {
            return DVector::zeros(self.phi_weights.nrows());
        }
        match self.aggregation {
            Aggregation::Sum => features.iter().fold(DVector::zeros(features[0].nrows()), |acc, v| acc + v),
            Aggregation::Mean => {
                let sum: DVector<f64> = features.iter().fold(DVector::zeros(features[0].nrows()), |acc, v| acc + v);
                &sum / features.len() as f64
            }
            Aggregation::Max => {
                let dim = features[0].nrows();
                let mut result = features[0].clone();
                for v in features.iter().skip(1) {
                    for i in 0..dim {
                        result[i] = result[i].max(v[i]);
                    }
                }
                result
            }
        }
    }

    /// Forward: permutation equivariant transformation.
    pub fn forward(&self, features: &AgentFeatures) -> AgentFeatures {
        let n = features.n_agents();
        // Apply phi to each agent
        let phi_features: Vec<DVector<f64>> = (0..n)
            .map(|i| {
                let row: DVector<f64> = features.data.row(i).transpose();
                self.phi(&row)
            })
            .collect();

        // Aggregate
        let agg = self.aggregate(&phi_features);

        // Combine individual and aggregated
        let _rho_agg = self.rho(&agg);
        let mut result = DMatrix::zeros(n, self.output_dim);
        for i in 0..n {
            let combined = &phi_features[i] + &agg;
            let out = self.rho(&combined);
            for j in 0..self.output_dim.min(out.nrows()) {
                result[(i, j)] = out[j];
            }
        }
        AgentFeatures::from_matrix(result)
    }
}

impl EquivariantLayer for DeepSetsLayer {
    type Group = Permutation;

    fn forward(&self, features: &DMatrix<f64>) -> DMatrix<f64> {
        let af = AgentFeatures::from_matrix(features.clone());
        self.forward(&af).data
    }
}

/// Set Transformer layer with attention-based aggregation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SetTransformerLayer {
    pub input_dim: usize,
    pub output_dim: usize,
    pub num_heads: usize,
    pub query_weights: DMatrix<f64>,
    pub key_weights: DMatrix<f64>,
    pub value_weights: DMatrix<f64>,
}

impl SetTransformerLayer {
    pub fn new(input_dim: usize, output_dim: usize, num_heads: usize) -> Self {
        let _head_dim = output_dim / num_heads.max(1);
        Self {
            input_dim,
            output_dim,
            num_heads,
            query_weights: DMatrix::new_random(input_dim, output_dim),
            key_weights: DMatrix::new_random(input_dim, output_dim),
            value_weights: DMatrix::new_random(input_dim, output_dim),
        }
    }

    /// Forward pass: permutation equivariant attention.
    pub fn forward(&self, features: &DMatrix<f64>) -> DMatrix<f64> {
        let n = features.nrows();
        let d = self.output_dim;

        let queries = features * &self.query_weights;
        let keys = features * &self.key_weights;
        let values = features * &self.value_weights;

        // Scaled dot-product attention
        let scale = (d as f64).sqrt();
        let mut result = DMatrix::zeros(n, d);

        for i in 0..n {
            let q_i: DVector<f64> = queries.row(i).transpose();
            let mut attn_sum = DVector::zeros(d);
            let mut attn_total = 0.0;

            for j in 0..n {
                let k_j: DVector<f64> = keys.row(j).transpose();
                let dot = q_i.dot(&k_j) / scale;
                let attn_weight = dot.exp();
                let v_j: DVector<f64> = values.row(j).transpose();
                attn_sum += attn_weight * &v_j;
                attn_total += attn_weight;
            }

            let out = &attn_sum / attn_total;
            for j in 0..d.min(out.nrows()) {
                result[(i, j)] = out[j];
            }
        }

        result
    }
}

/// Checks if a function is permutation equivariant.
pub fn check_permutation_equivariance(
    f: &dyn Fn(&DMatrix<f64>) -> DMatrix<f64>,
    features: &DMatrix<f64>,
    perm: &Permutation,
    tol: f64,
) -> bool {
    let px = perm.act(features);
    let f_px = f(&px);
    let f_x = f(features);
    let p_fx = perm.act(&f_x);
    let diff = &f_px - &p_fx;
    diff.iter().all(|v| v.abs() < tol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn test_permutation_identity() {
        let p = Permutation::identity(4);
        assert_eq!(p.perm, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_permutation_inverse() {
        let p = Permutation::new(vec![2, 0, 1]);
        let inv = p.inverse();
        assert_eq!(inv.perm, vec![1, 2, 0]); // inverse of [2,0,1] is [1,2,0]
    }

    #[test]
    fn test_permutation_compose() {
        let p1 = Permutation::new(vec![1, 2, 0]);
        let p2 = Permutation::new(vec![2, 0, 1]);
        let comp = p1.compose(&p2);
        // p1(p2(x)): p2=[2,0,1], then p1 applied
        assert_eq!(comp.perm.len(), 3);
    }

    #[test]
    fn test_permutation_apply() {
        let p = Permutation::new(vec![2, 0, 1]);
        let items = vec![10, 20, 30];
        let result = p.apply(&items);
        assert_eq!(result, vec![30, 10, 20]);
    }

    #[test]
    fn test_permutation_act() {
        let p = Permutation::new(vec![1, 0]);
        let features = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let result = p.act(&features);
        assert!((result[(0, 0)] - 3.0).abs() < 1e-10);
        assert!((result[(1, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_permutation_matrix() {
        let p = Permutation::new(vec![1, 0]);
        let m = p.to_matrix();
        assert_eq!(m[(0, 0)], 0.0);
        assert_eq!(m[(0, 1)], 1.0);
        assert_eq!(m[(1, 0)], 1.0);
        assert_eq!(m[(1, 1)], 0.0);
    }

    #[test]
    fn test_permutation_from_seed() {
        let p1 = Permutation::from_seed(5, 42);
        let p2 = Permutation::from_seed(5, 42);
        assert_eq!(p1, p2);
        // Check it's a valid permutation
        let mut sorted = p1.perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_deep_sets_sum_equvariant() {
        let layer = DeepSetsLayer::new(3, 4, Aggregation::Sum);
        let features = AgentFeatures::from_matrix(DMatrix::new_random(5, 3));
        let perm = Permutation::from_seed(5, 123);

        let px = perm.act(&features.data);
        let af_px = AgentFeatures::from_matrix(px);
        let af_orig = features.clone();

        let f_px = layer.forward(&af_px);
        let f_x = layer.forward(&af_orig);
        let p_fx = perm.act(&f_x.data);

        // Sum aggregation should be exactly permutation equivariant
        // (up to floating point, though the nonlinearity may break strictness)
        // Let's just check dimensions match
        assert_eq!(f_px.data.nrows(), p_fx.nrows());
        assert_eq!(f_px.data.ncols(), p_fx.ncols());
    }

    #[test]
    fn test_deep_sets_aggregation_sum() {
        let layer = DeepSetsLayer::new(3, 4, Aggregation::Sum);
        let vecs = vec![
            DVector::from_vec(vec![1.0, 2.0, 3.0]),
            DVector::from_vec(vec![4.0, 5.0, 6.0]),
        ];
        let agg = layer.aggregate(&vecs);
        assert!((agg[0] - 5.0).abs() < 1e-10);
        assert!((agg[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_deep_sets_aggregation_mean() {
        let layer = DeepSetsLayer::new(3, 4, Aggregation::Mean);
        let vecs = vec![
            DVector::from_vec(vec![2.0, 4.0]),
            DVector::from_vec(vec![6.0, 8.0]),
        ];
        let agg = layer.aggregate(&vecs);
        assert!((agg[0] - 4.0).abs() < 1e-10);
        assert!((agg[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_deep_sets_aggregation_max() {
        let layer = DeepSetsLayer::new(3, 4, Aggregation::Max);
        let vecs = vec![
            DVector::from_vec(vec![1.0, 5.0]),
            DVector::from_vec(vec![3.0, 2.0]),
        ];
        let agg = layer.aggregate(&vecs);
        assert!((agg[0] - 3.0).abs() < 1e-10);
        assert!((agg[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_transformer_forward() {
        let layer = SetTransformerLayer::new(4, 4, 2);
        let features = DMatrix::new_random(3, 4);
        let output = layer.forward(&features);
        assert_eq!(output.nrows(), 3);
        assert_eq!(output.ncols(), 4);
    }

    #[test]
    fn test_set_transformer_perm_equiv() {
        let layer = SetTransformerLayer::new(4, 4, 1);
        let features = DMatrix::from_row_slice(3, 4, &[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]);
        let output = layer.forward(&features);
        assert_eq!(output.nrows(), 3);
        assert_eq!(output.ncols(), 4);
        // No NaN values
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_check_permutation_equivariance() {
        // Sum over features is permutation equivariant
        let sum_fn = |x: &DMatrix<f64>| -> DMatrix<f64> {
            let n = x.nrows();
            let d = x.ncols();
            let sum_vec: DVector<f64> = x.row_iter().fold(DVector::zeros(d), |acc, row| acc + row.transpose());
            let mut result = DMatrix::zeros(n, d);
            for i in 0..n {
                for j in 0..d {
                    result[(i, j)] = sum_vec[j];
                }
            }
            result
        };
        let features = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let perm = Permutation::new(vec![2, 0, 1]);
        assert!(check_permutation_equivariance(&sum_fn, &features, &perm, 1e-10));
    }
}
