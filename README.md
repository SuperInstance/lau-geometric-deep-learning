# lau-geometric-deep-learning

**Bronstein et al.'s Geometric Deep Learning framework — the 5 symmetries applied to agent systems.**

Geometric deep learning extends convolutional networks to non-Euclidean domains (graphs, manifolds, groups) by enforcing equivariance: if you transform the input by a symmetry, the output transforms the same way. This crate implements permutation, translation, rotation, scale, time, and gauge equivariance for agent feature processing.

132 tests · MIT license · pure Rust · zero unsafe

---

## What This Does

| Module | Symmetry | What you get |
|---|---|---|
| `core` | Framework foundations | `GroupAction` trait, `EquivariantLayer` trait, `AgentFeatures`, `AgentGraph` |
| `permutation` | Agent order doesn't matter | Deep Sets (φ/ρ architecture), permutation check, set aggregation |
| `translation` | Shift-invariant features | Weight-shared linear layers, circular convolution |
| `rotation` | SO(n) equivariance | 2D/3D rotation actions, rotation-invariant features (distances, angles) |
| `scale` | Multi-resolution features | Normalization, log-scale, multi-scale processing |
| `time_equiv` | Temporal shift equivariance | Circular shift, temporal convolution |
| `spectral` | Graph Fourier domain | Chebyshev spectral filters, low-pass / high-pass / band-pass |
| `spatial` | Message passing on graphs | MPNN layers with sum/mean/max aggregation |
| `gauge` | Local coordinate changes | Per-agent gauge matrices, gauge-equivariant layers |
| `group_conv` | General group convolution | Cyclic and dihedral groups, group-theoretic convolution |
| `agent` | Unified equivariant model | `EquivariantAgentModel` combining all symmetries |
| `universal` | Universal approximation | Deep Sets as universal approximators for permutation-equivariant functions |

---

## Key Idea

> A function f is **equivariant** to a group G if f(g · x) = g · f(x) for all g ∈ G. By building this constraint into the architecture, you get generalization for free — the network cannot learn to depend on things that the symmetry says shouldn't matter.

For agent systems, the most important symmetry is **permutation**: agents have no canonical ordering. A Deep Set (φ network + aggregation + ρ network) is the universal permutation-equivariant architecture. On top of that, you can add translation equivariance (agents on a lattice), rotation equivariance (agents in physical space), scale equivariance (multi-resolution), and gauge equivariance (features in local coordinate systems).

---

## Install

```toml
[dependencies]
lau-geometric-deep-learning = { git = "https://github.com/SuperInstance/lau-geometric-deep-learning" }
```

Requires Rust 2021 edition. Dependencies: `nalgebra`, `serde`, `serde_json`.

---

## Quick Start

### Permutation-equivariant processing (Deep Sets)

```rust
use lau_geometric_deep_learning::{
    AgentFeatures, DeepSetsLayer, Permutation, Aggregation,
};
use nalgebra::DMatrix;

// 5 agents, 3 features each
let features = AgentFeatures::new(DMatrix::from_element(5, 3, 1.0));
let layer = DeepSetsLayer::new(3, 8, Aggregation::Sum);
let output = layer.forward(&features);
// Output is the same no matter how you reorder the agents

let perm = Permutation::from_seed(5, 42);
assert!(layer.check_equivariance(&perm, &features, 1e-6));
```

### Spectral graph filtering

```rust
use lau_geometric_deep_learning::{AgentGraph, SpectralFilter};

// Build a graph with 10 agents
let graph = AgentGraph::random(10, 0.3);
let features = AgentFeatures::new(DMatrix::from_element(10, 4, 1.0));

let lowpass = SpectralFilter::low_pass(5, 2.0);
let smoothed = lowpass.apply_exact(&graph, &features.data());
```

### Full equivariant model

```rust
use lau_geometric_deep_learning::{GeometricModelConfig, EquivariantAgentModel, SymmetryType};

let config = GeometricModelConfig {
    input_dim: 4,
    hidden_dim: 16,
    output_dim: 2,
    num_layers: 3,
    symmetries: vec![SymmetryType::Permutation],
    use_spatial: true,
    use_spectral: false,
};

let model = EquivariantAgentModel::new(config);
let features = AgentFeatures::new(DMatrix::new_random(8, 4));
let output = model.forward(&features);
```

---

## API Reference

### Core

**`GroupAction`** trait — any symmetry element that can act on features:
- `g.act(&features)` — apply the transformation.
- `g.inverse()` — inverse element.
- `g.compose(&other)` — group multiplication.
- `G::identity()` — the identity element.

**`EquivariantLayer`** trait — a layer that commutes with a group:
- `layer.forward(&features)` — standard forward pass.
- `layer.check_equivariance(&g, &features, tol)` — verify f(g·x) = g·f(x).

**`AgentFeatures`** — feature matrix (n_agents × feature_dim):
- `AgentFeatures::new(data)` — from a `DMatrix<f64>`.
- `features.n_agents()` / `features.feature_dim()`.
- `features.data()` — access the underlying matrix.

**`AgentGraph`** — adjacency structure for agents:
- `AgentGraph::random(n, edge_prob)` — Erdős–Rényi.
- `graph.adjacency_matrix()` / `graph.degree_matrix()`.
- `graph.normalized_laplacian()` — for spectral methods.

**`SymmetryType`** — enum: `Permutation`, `Translation { dim }`, `Rotation { dim }`, `Scale`, `Time`, `Gauge { fiber_dim }`, `GroupConv { order }`.

### Permutation

**`Permutation`** — a permutation of agent indices:
- `Permutation::new(perm)` — from explicit index list.
- `Permutation::identity(n)` — the identity.
- `Permutation::from_seed(n, seed)` — deterministic random.
- `p.inverse()` / `p.compose(&other)`.
- `p.apply_to_matrix(&m)` — permute rows.

**`DeepSetsLayer`** — the universal permutation-equivariant layer:
- `DeepSetsLayer::new(input_dim, output_dim, aggregation)`.
- Aggregation: `Sum`, `Mean`, `Max`.
- `layer.forward(&features)` — φ network, aggregate, ρ network.

**`Aggregation`** — `Sum`, `Mean`, `Max`.

### Translation

**`Translation`** — a shift in feature space:
- `Translation::new(shift)` — from a `DVector<f64>`.
- `Translation::zero(dim)` — no shift.
- Implements `GroupAction`: adds shift to each row.

**`TranslationEquivLinear`** — weight-shared linear layer:
- `TranslationEquivLinear::new(input_dim, output_dim)`.
- `layer.forward(&features)` — same weights applied to every agent.

### Rotation

**`RotationAction`** — 2D or 3D rotation:
- `RotationAction::rot2(angle)` — 2D rotation by angle.
- `RotationAction::rot3_from_axis_angle(&axis, angle)` — 3D axis-angle.
- `r.rotate_positions(&positions)` — apply to position matrix.

**`RotationInvariantFeatures`** — extract rotation-invariant quantities:
- Pairwise distances.
- Angles between agent triplets.

### Scale

**`ScaleAction`** — a scaling factor:
- `ScaleAction::new(factor)`.
- Implements `GroupAction`: multiplies all features by factor.

**`ScaleEquivLayer`** — scale-equivariant layer:
- `ScaleMode::Normalize` — normalize by L2 norm.
- `ScaleMode::LogScale` — log-transform.
- `ScaleMode::MultiScale { scales }` — process at multiple scales.

### Time

**`TimeShift`** — circular shift of the time axis:
- `TimeShift::new(steps)` — shift by `steps` positions.
- Implements `GroupAction`: circular index permutation.

**`TemporalConvLayer`** — temporal convolution (shift-equivariant):
- `TemporalConvLayer::new(kernel_size, input_channels, output_channels)`.
- `layer.forward(&features)` — 1D convolution with circular padding.

### Spectral

**`SpectralFilter`** — convolution in the graph Fourier domain:
- `SpectralFilter::new(order)` — random Chebyshev coefficients.
- `SpectralFilter::low_pass(order, cutoff)` — smoothing filter.
- `SpectralFilter::high_pass(order, cutoff)` — edge detection.
- `SpectralFilter::band_pass(order, low, high)` — band-pass.
- `filter.apply_exact(&graph, &features)` — full eigendecomposition.
- `filter.apply_chebyshev(&graph, &features)` — Chebyshev approximation.

### Spatial

**`MessagePassingLayer`** — message-passing neural network:
- `MessagePassingLayer::new(input_dim, output_dim, aggregation)`.
- `layer.message(&h_i, &h_j)` — compute message between agents.
- `layer.aggregate(&messages)` — combine neighbor messages.
- `layer.update(&h, &aggregated)` — update agent state.
- `layer.forward(&graph, &features)` — full MPNN step.

### Gauge

**`GaugeTransformation`** — per-agent coordinate change:
- `GaugeTransformation::identity(n_agents, fiber_dim)`.
- `GaugeTransformation::random_orthogonal(n_agents, fiber_dim, seed)`.

**`GaugeEquivLayer`** — gauge-equivariant processing:
- `layer.forward(&features, &graph)` — parallel transport + linear + inverse transport.

### Group Convolution

**`FiniteGroup`** — a group by its multiplication table:
- `FiniteGroup::cyclic(n)` — Z_n.
- `FiniteGroup::dihedral(n)` — D_n (rotations + reflections of regular n-gon).

**`GroupConvLayer`** — general G-equivariant convolution:
- `GroupConvLayer::new(group, input_channels, output_channels)`.
- `layer.forward(&features)` — group convolution (f * g)(x) = Σ_{y∈G} f(y) g(y⁻¹x).

### Agent Model

**`EquivariantAgentModel`** — unified model combining selected symmetries:
- `EquivariantAgentModel::new(config)`.
- `model.forward(&features)` — full forward pass through all selected layers.
- `model.forward_with_graph(&features, &graph)` — for spatial/spectral layers.

**`GeometricModelConfig`**:
- `input_dim`, `hidden_dim`, `output_dim`, `num_layers`.
- `symmetries: Vec<SymmetryType>` — which equivariances to enforce.
- `use_spatial` — enable message passing.
- `use_spectral` — enable spectral filters.

### Universal Approximation

**`UniversalApproximator`** — deep permutation-equivariant network:
- `UniversalApproximator::new(input_dim, hidden_dim, output_dim, depth)`.
- `approx.forward(&features)`.
- `approx.empirical_error(&features, &target)` — measure approximation quality.

---

## How It Works

1. **Define the symmetry**: Choose which group(s) your data respects — permutation (no canonical ordering), translation (shift-invariant), rotation (orientation-invariant), etc.

2. **Build equivariant layers**: Each layer type enforces its symmetry by construction. A Deep Sets layer aggregates across all agents identically; a spectral filter convolves with the graph Laplacian eigenvectors; a temporal convolution uses circular padding.

3. **Verify equivariance**: Every layer implements `check_equivariance`, which directly tests f(g·x) ≈ g·f(x) numerically. Use this in tests to verify your architecture.

4. **Compose layers**: Stack equivariant layers — the composition of equivariant maps is equivariant. The `EquivariantAgentModel` handles this automatically.

5. **Process agent data**: Feed agent features through the model. The output respects all specified symmetries regardless of the learned weights.

---

## The Math

### Equivariance

A function f: X → Y is **G-equivariant** if f(g · x) = g · f(x) for all g ∈ G. If g · f(x) = f(x) (the output is unchanged), f is **G-invariant**. Equivariance is the natural notion for intermediate layers; invariance is for final outputs.

### Deep Sets (Permutation Equivariance)

**Theorem** (Zaheer et al., 2017): Any continuous permutation-equivariant function f: ℝ^{n×d} → ℝ^{n×d'} can be approximated by f(X)ᵢ = ρ(φ(xᵢ), AGG_{j} φ(xⱼ)) where φ, ρ are neural networks and AGG is sum, mean, or max.

### Graph Fourier Transform

For a graph with Laplacian L = UΛUᵀ, the **graph Fourier transform** of a signal x is x̂ = Uᵀx. Convolution in the spectral domain is ĥ ⊙ x̂ (element-wise multiplication). Spectral filters learn coefficients in the eigenbasis.

### Chebyshev Approximation

Computing the full eigendecomposition is O(n³). Chebyshev spectral filters approximate g(Λ) ≈ Σₖ θₖ Tₖ(Λ̃) where Tₖ are Chebyshev polynomials and Λ̃ = 2Λ/λ_max − I. This costs O(k|E|) — linear in edges.

### Message Passing

An MPNN layer updates each node by: hᵢ' = UPDATE(hᵢ, AGG_{j∈N(i)} MESSAGE(hᵢ, hⱼ)). This is permutation-equivariant by construction (AGG is symmetric). Universal for functions on graphs with sufficient depth.

### Gauge Equivariance

On a manifold with local frames (gauges) at each point, a feature vector v transforms as v → gᵢv when the gauge at point i changes. A gauge-equivariant layer applies: gᵢ⁻¹ L (gᵢ vᵢ) — transform to a canonical frame, apply the linear map, transform back. This ensures the output doesn't depend on the arbitrary choice of local coordinates.

### Group Convolution

For a finite group G, the convolution (f * ψ)(x) = Σ_{y∈G} f(y) ψ(y⁻¹x) is G-equivariant. This generalizes standard convolution (translation group) to any group with a known multiplication table.

---

## License

MIT
