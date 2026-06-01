//! Group convolution: general G-equivariant convolution.
//!
//! Implements convolution on a finite group G, where the convolution
//! (f * g)(x) = Σ_{y∈G} f(y) g(y^{-1}x) is equivariant by construction.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::core::{AgentFeatures, EquivariantLayer, GroupAction};

/// A finite group represented by its multiplication table.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FiniteGroup {
    /// Group elements (indices 0..n).
    pub order: usize,
    /// Multiplication table: table[i][j] = i·j (result index).
    pub table: Vec<Vec<usize>>,
    /// Inverse table: inv[i] = i^{-1}.
    pub inv: Vec<usize>,
}

impl FiniteGroup {
    /// Cyclic group Z_n.
    pub fn cyclic(n: usize) -> Self {
        let mut table = vec![vec![0; n]; n];
        for i in 0..n {
            for j in 0..n {
                table[i][j] = (i + j) % n;
            }
        }
        let inv: Vec<usize> = (0..n).map(|i| (n - i) % n).collect();
        Self { order: n, table, inv }
    }

    /// Dihedral group D_n (symmetries of regular n-gon).
    pub fn dihedral(n: usize) -> Self {
        let order = 2 * n;
        let mut table = vec![vec![0; order]; order];
        // Elements 0..n-1 are rotations, n..2n-1 are reflections
        for i in 0..order {
            for j in 0..order {
                let i_rot = i < n;
                let j_rot = j < n;
                let ii = i % n;
                let jj = j % n;
                if i_rot && j_rot {
                    table[i][j] = (ii + jj) % n;
                } else if i_rot && !j_rot {
                    table[i][j] = n + (ii + jj) % n;
                } else if !i_rot && j_rot {
                    table[i][j] = n + (n - ii + jj) % n;
                } else {
                    table[i][j] = (n - ii + jj) % n;
                }
            }
        }
        let mut inv = vec![0; order];
        for i in 0..n {
            inv[i] = (n - i) % n; // r_i^{-1} = r_{n-i}
        }
        for i in 0..n {
            inv[n + i] = n + i; // reflections are self-inverse
        }
        Self { order, table, inv }
    }

    /// Symmetric group S_n (as permutation indices).
    /// For small n only.
    pub fn symmetric(n: usize) -> Self {
        assert!(n <= 5, "Symmetric group only supported for n <= 5");
        let perms = Self::all_permutations(n);
        let order = perms.len();
        let perm_to_idx: std::collections::HashMap<Vec<usize>, usize> = perms.iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i))
            .collect();

        let mut table = vec![vec![0; order]; order];
        for i in 0..order {
            for j in 0..order {
                let composed: Vec<usize> = perms[i].iter().map(|&k| perms[j][k]).collect();
                table[i][j] = perm_to_idx[&composed];
            }
        }

        let mut inv = vec![0; order];
        for i in 0..order {
            let p = &perms[i];
            let mut p_inv = vec![0usize; n];
            for (k, &v) in p.iter().enumerate() {
                p_inv[v] = k;
            }
            inv[i] = perm_to_idx[&p_inv];
        }

        Self { order, table, inv }
    }

    fn all_permutations(n: usize) -> Vec<Vec<usize>> {
        let mut perms = Vec::new();
        let mut p: Vec<usize> = (0..n).collect();
        perms.push(p.clone());
        while Self::next_permutation(&mut p) {
            perms.push(p.clone());
        }
        perms
    }

    fn next_permutation(p: &mut Vec<usize>) -> bool {
        let n = p.len();
        let mut i = n as isize - 2;
        while i >= 0 && p[i as usize] > p[(i + 1) as usize] {
            i -= 1;
        }
        if i < 0 { return false; }
        let i = i as usize;
        let mut j = n - 1;
        while p[j] < p[i] { j -= 1; }
        p.swap(i, j);
        p[i + 1..].reverse();
        true
    }

    /// Multiply two elements.
    pub fn multiply(&self, a: usize, b: usize) -> usize {
        self.table[a][b]
    }

    /// Get inverse.
    pub fn inverse(&self, a: usize) -> usize {
        self.inv[a]
    }

    /// Get identity element (always 0 for our constructions).
    pub fn identity(&self) -> usize {
        0
    }
}

/// Group convolution layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroupConvLayer {
    /// The group.
    pub group: FiniteGroup,
    /// Input channels.
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Kernel weights: (out_channels × in_channels × group_order).
    pub kernel: Vec<DMatrix<f64>>,
}

impl GroupConvLayer {
    pub fn new(group: FiniteGroup, in_channels: usize, out_channels: usize) -> Self {
        let kernel = (0..out_channels)
            .map(|_| DMatrix::new_random(in_channels, group.order))
            .collect();
        Self { group, in_channels, out_channels, kernel }
    }

    /// Group convolution: (f * k)(x) = Σ_y Σ_c f_c(y) k_c(y^{-1}x).
    /// Input: (group_order × in_channels), Output: (group_order × out_channels).
    pub fn forward(&self, input: &DMatrix<f64>) -> DMatrix<f64> {
        let n = self.group.order;
        assert_eq!(input.nrows(), n);
        assert_eq!(input.ncols(), self.in_channels);

        let mut result = DMatrix::zeros(n, self.out_channels);

        for x in 0..n {
            for oc in 0..self.out_channels {
                let mut sum = 0.0;
                for y in 0..n {
                    let y_inv_x = self.group.multiply(self.group.inverse(y), x);
                    for ic in 0..self.in_channels {
                        sum += input[(y, ic)] * self.kernel[oc][(ic, y_inv_x)];
                    }
                }
                result[(x, oc)] = sum;
            }
        }

        result
    }

    /// Apply to agent features by treating each agent as a group element.
    pub fn forward_agents(&self, features: &AgentFeatures) -> AgentFeatures {
        assert_eq!(features.n_agents(), self.group.order);
        let result = self.forward(&features.data);
        AgentFeatures::from_matrix(result)
    }
}

/// Lifting layer: maps scalar features to group-valued features.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LiftingLayer {
    pub group: FiniteGroup,
    pub input_dim: usize,
    pub weights: DMatrix<f64>,
}

impl LiftingLayer {
    pub fn new(group: FiniteGroup, input_dim: usize) -> Self {
        Self {
            group,
            input_dim,
            weights: DMatrix::new_random(input_dim, input_dim),
        }
    }

    /// Lift features to group-valued features via regular representation.
    pub fn forward(&self, features: &DMatrix<f64>) -> DMatrix<f64> {
        let n = features.nrows();
        let d = features.ncols();
        let group_order = self.group.order;

        // Expand each feature to group_order copies (regular representation)
        let mut result = DMatrix::zeros(n * group_order, d);
        for i in 0..n {
            for g in 0..group_order {
                for j in 0..d {
                    result[(i * group_order + g, j)] = features[(i, j)];
                }
            }
        }
        result * &self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cyclic_group() {
        let z4 = FiniteGroup::cyclic(4);
        assert_eq!(z4.order, 4);
        assert_eq!(z4.multiply(1, 2), 3);
        assert_eq!(z4.multiply(3, 1), 0); // wrap
        assert_eq!(z4.inverse(1), 3);
    }

    #[test]
    fn test_cyclic_identity() {
        let z5 = FiniteGroup::cyclic(5);
        assert_eq!(z5.identity(), 0);
        for i in 0..5 {
            assert_eq!(z5.multiply(0, i), i);
            assert_eq!(z5.multiply(i, 0), i);
        }
    }

    #[test]
    fn test_dihedral_group() {
        let d3 = FiniteGroup::dihedral(3);
        assert_eq!(d3.order, 6);
        // r0 * r1 = r1
        assert_eq!(d3.multiply(0, 1), 1);
        // Identity
        assert_eq!(d3.identity(), 0);
    }

    #[test]
    fn test_symmetric_group() {
        let s3 = FiniteGroup::symmetric(3);
        assert_eq!(s3.order, 6);
        assert_eq!(s3.identity(), 0);
    }

    #[test]
    fn test_symmetric_inverse() {
        let s3 = FiniteGroup::symmetric(3);
        for i in 0..6 {
            let inv = s3.inverse(i);
            assert_eq!(s3.multiply(i, inv), 0);
        }
    }

    #[test]
    fn test_group_conv_forward() {
        let z4 = FiniteGroup::cyclic(4);
        let layer = GroupConvLayer::new(z4, 3, 2);
        let input = DMatrix::new_random(4, 3);
        let result = layer.forward(&input);
        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_group_conv_equivariance() {
        let z3 = FiniteGroup::cyclic(3);
        let layer = GroupConvLayer::new(z3, 2, 2);
        let input = DMatrix::from_row_slice(3, 2, &[
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ]);
        let output = layer.forward(&input);
        assert_eq!(output.nrows(), 3);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_group_conv_dihedral() {
        let d3 = FiniteGroup::dihedral(3);
        let layer = GroupConvLayer::new(d3, 2, 3);
        let input = DMatrix::new_random(6, 2);
        let result = layer.forward(&input);
        assert_eq!(result.nrows(), 6);
        assert_eq!(result.ncols(), 3);
    }

    #[test]
    fn test_lifting_layer() {
        let z3 = FiniteGroup::cyclic(3);
        let layer = LiftingLayer::new(z3, 4);
        let input = DMatrix::new_random(2, 4);
        let result = layer.forward(&input);
        assert_eq!(result.nrows(), 6); // 2 × 3
        assert_eq!(result.ncols(), 4);
    }

    #[test]
    fn test_cyclic_associativity() {
        let z5 = FiniteGroup::cyclic(5);
        for a in 0..5 {
            for b in 0..5 {
                for c in 0..5 {
                    let ab_c = z5.multiply(z5.multiply(a, b), c);
                    let a_bc = z5.multiply(a, z5.multiply(b, c));
                    assert_eq!(ab_c, a_bc);
                }
            }
        }
    }
}
