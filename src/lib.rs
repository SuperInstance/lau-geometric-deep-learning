//! # lau-geometric-deep-learning
//!
//! Bronstein et al.'s Geometric Deep Learning framework — the 5 symmetries
//! (ℤ² equivariance, gauge equivariance, permutation equivariance, scale equivariance,
//! time equivariance) applied to agent systems.
//!
//! ## Core Concepts
//!
//! - **Equivariant layers**: functions f where f(g·x) = g·f(x) for symmetry group G
//! - **Permutation equivariance**: agent order doesn't matter (connect to sheaf-neural)
//! - **Translation equivariance**: shift-invariant agent features
//! - **Rotation equivariance**: SO(n) equivariant features on agent manifolds
//! - **Scale equivariance**: multi-resolution agent features
//! - **Spectral filters**: convolution in frequency domain via graph Fourier transform
//! - **Spatial filters**: message passing with geometric weight sharing
//! - **Gauge equivariance**: features that transform correctly under change of local coordinates
//! - **Group convolution**: general G-equivariant convolution

pub mod core;
pub mod permutation;
pub mod translation;
pub mod rotation;
pub mod scale;
pub mod time_equiv;
pub mod spectral;
pub mod spatial;
pub mod gauge;
pub mod group_conv;
pub mod agent;
pub mod universal;
pub mod prelude;

pub use prelude::*;
