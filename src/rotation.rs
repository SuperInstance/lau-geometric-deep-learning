//! Rotation equivariance: SO(n) equivariant features on agent manifolds.
//!
//! Features that are invariant or equivariant to rotations of the agent
//! configuration in Euclidean space.

use nalgebra::{DMatrix, DVector, Rotation2, Rotation3};
use serde::{Deserialize, Serialize};

use crate::core::{AgentFeatures, GroupAction};

/// A rotation action in 2D or 3D.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RotationAction {
    Rot2 { angle: f64 },
    Rot3 { matrix: [[f64; 3]; 3] },
}

impl RotationAction {
    pub fn rot2(angle: f64) -> Self {
        Self::Rot2 { angle }
    }

    pub fn rot3_from_axis_angle(axis: &[f64; 3], angle: f64) -> Self {
        let ax = nalgebra::Vector3::new(axis[0], axis[1], axis[2]);
        let unit_ax = nalgebra::Unit::new_normalize(ax);
        let r = Rotation3::from_axis_angle(&unit_ax, angle);
        Self::Rot3 {
            matrix: [
                [r[(0, 0)], r[(0, 1)], r[(0, 2)]],
                [r[(1, 0)], r[(1, 1)], r[(1, 2)]],
                [r[(2, 0)], r[(2, 1)], r[(2, 2)]],
            ],
        }
    }

    /// Apply rotation to position matrix (n × d).
    pub fn rotate_positions(&self, positions: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            RotationAction::Rot2 { angle } => {
                let r = Rotation2::new(*angle);
                let mut result = positions.clone();
                for i in 0..positions.nrows() {
                    let pos = nalgebra::Vector2::new(positions[(i, 0)], positions[(i, 1)]);
                    let rotated = r * pos;
                    result[(i, 0)] = rotated[0];
                    result[(i, 1)] = rotated[1];
                }
                result
            }
            RotationAction::Rot3 { matrix } => {
                let r = Rotation3::from_matrix(&nalgebra::Matrix3::new(
                    matrix[0][0], matrix[0][1], matrix[0][2],
                    matrix[1][0], matrix[1][1], matrix[1][2],
                    matrix[2][0], matrix[2][1], matrix[2][2],
                ));
                let mut result = positions.clone();
                for i in 0..positions.nrows() {
                    let pos = nalgebra::Vector3::new(
                        positions[(i, 0)],
                        positions.get((i, 1)).copied().unwrap_or(0.0),
                        positions.get((i, 2)).copied().unwrap_or(0.0),
                    );
                    let rotated = r * pos;
                    for j in 0..positions.ncols().min(3) {
                        result[(i, j)] = rotated[j];
                    }
                }
                result
            }
        }
    }
}

impl GroupAction for RotationAction {
    fn act(&self, features: &DMatrix<f64>) -> DMatrix<f64> {
        self.rotate_positions(features)
    }

    fn inverse(&self) -> Self {
        match self {
            RotationAction::Rot2 { angle } => Self::Rot2 { angle: -angle },
            RotationAction::Rot3 { matrix } => {
                // Transpose for inverse rotation
                Self::Rot3 {
                    matrix: [
                        [matrix[0][0], matrix[1][0], matrix[2][0]],
                        [matrix[0][1], matrix[1][1], matrix[2][1]],
                        [matrix[0][2], matrix[1][2], matrix[2][2]],
                    ],
                }
            }
        }
    }

    fn compose(&self, other: &Self) -> Self {
        match (self, other) {
            (RotationAction::Rot2 { angle: a }, RotationAction::Rot2 { angle: b }) => {
                Self::Rot2 { angle: a + b }
            }
            _ => {
                // General case: apply sequentially
                let m = self.to_matrix_rep() * other.to_matrix_rep();
                if m.nrows() == 2 {
                    let angle = m[(0, 1)].asin();
                    Self::Rot2 { angle }
                } else {
                    Self::Rot3 {
                        matrix: [
                            [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
                            [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
                            [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
                        ],
                    }
                }
            }
        }
    }

    fn identity() -> Self {
        Self::Rot2 { angle: 0.0 }
    }
}

impl RotationAction {
    fn to_matrix_rep(&self) -> DMatrix<f64> {
        match self {
            RotationAction::Rot2 { angle } => {
                let c = angle.cos();
                let s = angle.sin();
                DMatrix::from_row_slice(2, 2, &[c, -s, s, c])
            }
            RotationAction::Rot3 { matrix } => {
                DMatrix::from_row_slice(3, 3, &[
                    matrix[0][0], matrix[0][1], matrix[0][2],
                    matrix[1][0], matrix[1][1], matrix[1][2],
                    matrix[2][0], matrix[2][1], matrix[2][2],
                ])
            }
        }
    }
}

/// Invariant features under rotation: distances, angles, triple products.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RotationInvariantFeatures {
    pub include_distances: bool,
    pub include_angles: bool,
    pub include_areas: bool,
}

impl Default for RotationInvariantFeatures {
    fn default() -> Self {
        Self {
            include_distances: true,
            include_angles: true,
            include_areas: false,
        }
    }
}

impl RotationInvariantFeatures {
    /// Compute rotation-invariant features from positions.
    /// Returns a feature vector for each agent based on its neighborhood.
    pub fn compute(&self, positions: &DMatrix<f64>) -> AgentFeatures {
        let n = positions.nrows();
        let mut all_features = Vec::new();

        for i in 0..n {
            let mut features = Vec::new();

            if self.include_distances {
                // Distances to all other agents
                for j in 0..n {
                    if i != j {
                        let dist = self.euclidean_distance(positions, i, j);
                        features.push(dist);
                    }
                }
            }

            if self.include_angles && n >= 3 {
                // Angles formed with pairs of other agents
                let others: Vec<usize> = (0..n).filter(|&j| j != i).collect();
                if others.len() >= 2 {
                    for k in 0..others.len() - 1 {
                        let angle = self.angle_at(positions, others[k], i, others[k + 1]);
                        features.push(angle.cos()); // Use cos for stability
                    }
                }
            }

            all_features.push(features);
        }

        let max_dim = all_features.iter().map(|f| f.len()).max().unwrap_or(0);
        let mut data = DMatrix::zeros(n, max_dim);
        for (i, feats) in all_features.iter().enumerate() {
            for (j, v) in feats.iter().enumerate() {
                data[(i, j)] = *v;
            }
        }
        AgentFeatures::from_matrix(data)
    }

    fn euclidean_distance(&self, positions: &DMatrix<f64>, i: usize, j: usize) -> f64 {
        let d = positions.ncols();
        let mut sum = 0.0;
        for k in 0..d {
            let diff = positions[(i, k)] - positions[(j, k)];
            sum += diff * diff;
        }
        sum.sqrt()
    }

    fn angle_at(&self, positions: &DMatrix<f64>, a: usize, b: usize, c: usize) -> f64 {
        let d = positions.ncols();
        let mut ba = vec![0.0; d];
        let mut bc = vec![0.0; d];
        for k in 0..d {
            ba[k] = positions[(a, k)] - positions[(b, k)];
            bc[k] = positions[(c, k)] - positions[(b, k)];
        }
        let dot: f64 = ba.iter().zip(bc.iter()).map(|(a, b)| a * b).sum();
        let mag_ba: f64 = ba.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mag_bc: f64 = bc.iter().map(|v| v * v).sum::<f64>().sqrt();
        if mag_ba > 0.0 && mag_bc > 0.0 {
            (dot / (mag_ba * mag_bc)).clamp(-1.0, 1.0).acos()
        } else {
            0.0
        }
    }
}

/// SO(n)-equivariant layer using steerable features.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SONEquivariantLayer {
    pub input_dim: usize,
    pub output_dim: usize,
    pub spatial_dim: usize, // 2 or 3
    pub weights: DMatrix<f64>,
}

impl SONEquivariantLayer {
    pub fn new(input_dim: usize, output_dim: usize, spatial_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            spatial_dim,
            weights: DMatrix::new_random(output_dim, input_dim),
        }
    }

    /// Forward: compute equivariant features from scalar invariants.
    pub fn forward(&self, positions: &DMatrix<f64>, features: &AgentFeatures) -> AgentFeatures {
        let n = positions.nrows();
        // Compute invariant distances
        let mut inv_feats = Vec::new();
        for i in 0..n {
            let mut feats = Vec::new();
            for j in 0..n {
                if i != j {
                    let mut dist_sq = 0.0;
                    for k in 0..positions.ncols() {
                        let diff = positions[(i, k)] - positions[(j, k)];
                        dist_sq += diff * diff;
                    }
                    feats.push(dist_sq.sqrt());
                }
            }
            // Append original features
            for j in 0..features.feature_dim() {
                feats.push(features.data[(i, j)]);
            }
            inv_feats.push(feats);
        }

        let _total_dim = inv_feats.first().map(|f| f.len()).unwrap_or(0);
        let mut result = DMatrix::zeros(n, self.output_dim);
        for i in 0..n {
            let v = DVector::from_vec(inv_feats[i].clone());
            if v.nrows() == self.weights.ncols() {
                let out = &self.weights * &v;
                for j in 0..self.output_dim.min(out.nrows()) {
                    result[(i, j)] = out[j];
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
    fn test_rotation_2d_act() {
        let r = RotationAction::rot2(std::f64::consts::PI / 2.0); // 90 degrees
        let pos = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let result = r.act(&pos);
        assert!(result[(0, 0)].abs() < 1e-10); // x → ~0
        assert!((result[(0, 1)] - 1.0).abs() < 1e-10); // y → 1
    }

    #[test]
    fn test_rotation_2d_inverse() {
        let r = RotationAction::rot2(std::f64::consts::FRAC_PI_4);
        let inv = r.inverse();
        let pos = DMatrix::from_row_slice(1, 2, &[3.0, 4.0]);
        let rotated = r.act(&pos);
        let recovered = inv.act(&rotated);
        assert!((recovered[(0, 0)] - 3.0).abs() < 1e-10);
        assert!((recovered[(0, 1)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_2d_compose() {
        let r1 = RotationAction::rot2(std::f64::consts::FRAC_PI_4);
        let r2 = RotationAction::rot2(std::f64::consts::FRAC_PI_4);
        let comp = r1.compose(&r2);
        let pos = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let result = comp.act(&pos);
        // 90 degree rotation: (1,0) → (0,1)
        assert!(result[(0, 0)].abs() < 1e-10);
        assert!((result[(0, 1)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_3d() {
        let r = RotationAction::rot3_from_axis_angle(&[0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
        let pos = DMatrix::from_row_slice(1, 3, &[1.0, 0.0, 0.0]);
        let result = r.act(&pos);
        assert!(result[(0, 0)].abs() < 1e-10);
        assert!((result[(0, 1)] - 1.0).abs() < 1e-10);
        assert!(result[(0, 2)].abs() < 1e-10);
    }

    #[test]
    fn test_rotation_3d_inverse() {
        let r = RotationAction::rot3_from_axis_angle(&[1.0, 0.0, 0.0], 1.0);
        let inv = r.inverse();
        let pos = DMatrix::from_row_slice(1, 3, &[1.0, 2.0, 3.0]);
        let rotated = r.act(&pos);
        let recovered = inv.act(&rotated);
        for j in 0..3 {
            assert!((recovered[(0, j)] - pos[(0, j)]).abs() < 1e-8);
        }
    }

    #[test]
    fn test_rotation_invariant_distances() {
        let inv = RotationInvariantFeatures {
            include_distances: true,
            include_angles: false,
            include_areas: false,
        };
        let pos = DMatrix::from_row_slice(3, 2, &[
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
        ]);
        let features = inv.compute(&pos);
        assert_eq!(features.n_agents(), 3);
        // Distances should be rotation-invariant
        let r = RotationAction::rot2(0.7);
        let rotated_pos = r.act(&pos);
        let rotated_features = inv.compute(&rotated_pos);
        for i in 0..3 {
            for j in 0..features.feature_dim() {
                assert!((features.data[(i, j)] - rotated_features.data[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_rotation_invariant_angles() {
        let inv = RotationInvariantFeatures::default();
        let pos = DMatrix::from_row_slice(3, 2, &[
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
        ]);
        let features = inv.compute(&pos);
        assert_eq!(features.n_agents(), 3);
        assert!(features.feature_dim() > 0);
    }

    #[test]
    fn test_so_n_equivariant_layer() {
        let layer = SONEquivariantLayer::new(5, 3, 2);
        let pos = DMatrix::from_row_slice(3, 2, &[
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
        ]);
        let feats = AgentFeatures::from_matrix(DMatrix::new_random(3, 2));
        let result = layer.forward(&pos, &feats);
        assert_eq!(result.n_agents(), 3);
        assert_eq!(result.feature_dim(), 3);
    }
}
