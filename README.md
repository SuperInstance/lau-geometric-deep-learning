# lau-geometric-deep-learning

**Geometric Deep Learning for agent systems** — implements all 5 GDL symmetries (permutation, translation, rotation, scale, time) as composable equivariant layers, plus spectral filters, spatial message passing, gauge equivariance, group convolution, and the universal approximation theorem for equivariant networks.

## What This Does

This crate operationalizes Bronstein et al.'s Geometric Deep Learning blueprint: neural network layers that respect the symmetries of the underlying domain. For agent systems, this means:

- **Permutation equivariance** — agent order doesn't matter (Deep Sets architecture)
- **Translation equivariance** — shift-invariant features via weight sharing
- **Rotation equivariance** — SO(2)/SO(3) equivariant features from invariant distances and angles
- **Scale equivariance** — multi-resolution features via normalization, log-scale, or multi-scale pooling
- **Time equivariance** — temporal convolution and recurrence that respect time shifts

Plus the building blocks that tie everything together:

- **Spectral filters** — convolution in the graph frequency domain (exact eigendecomposition or Chebyshev approximation)
- **Spatial filters** — message passing neural networks with sum/mean/max aggregation and optional attention
- **Gauge equivariance** — features that transform correctly under local coordinate changes
- **Group convolution** — general G-equivariant convolution for any finite group (cyclic Z_n, dihedral D_n)
- **Universal approximation** — theoretical guarantee that Deep Sets can approximate any continuous permutation-equivariant function
- **Unified agent model** — compose all equivariant layers into a single `EquivariantAgentModel`

## Key Idea

An **equivariant layer** f satisfies f(g·x) = g·f(x) for all group elements g. This is a structural constraint — not learned, but built into the architecture — that gives you:

1. **Data efficiency** — the network doesn't waste capacity learning the symmetry
2. **Guaranteed generalization** — equivariance holds exactly, not approximately
3. **Modularity** — compose different symmetries by stacking equivariant layers

The core trait `GroupAction` defines how a symmetry group element acts on a feature matrix. The trait `EquivariantLayer` defines a layer that commutes with the group action. Each module implements these for a specific symmetry group.

## Install

```toml
[dependencies]
lau-geometric-deep-learning = "0.1.0"
```

## Quick Start

```rust
use lau_geometric_deep_learning::prelude::*;

// Build an agent graph
let mut graph = AgentGraph::new(4);
graph.add_edge(0, 1, 1.0);
graph.add_edge(1, 2, 1.0);
graph.add_edge(2, 3, 1.0);

// Create features
let features = AgentFeatures::from_matrix(DMatrix::new_random(4, 8));

// Permutation-equivariant Deep Sets layer
let perm_layer = DeepSetsLayer::new(8, 16, Aggregation::Sum);
let out = perm_layer.forward(&features);

// Spectral filter on the graph
let filter = SpectralFilter::low_pass(3, 2.0);
let filtered = filter.apply_exact(&graph, &out.data);

// Message passing with attention
let mp = GraphConvLayer::new(8, 16).with_attention();
let mp_out = mp.forward(&graph, &features);

// Full equivariant model
let config = GeometricModelConfig {
    input_dim: 8,
    hidden_dim: 16,
    output_dim: 4,
    num_layers: 3,
    symmetries: vec![SymmetryType::Permutation, SymmetryType::Scale],
    use_spectral: true,
    use_spatial: true,
};
let model = EquivariantAgentModel::new(config);
let result = model.forward(&graph, &features);
```

## API Reference

### `core` — Types and Traits

| Type / Trait | Description |
|---|---|
| `GroupAction` | Trait: `act(&features)`, `inverse()`, `compose(&other)`, `identity()`. Defines how a group element transforms features. |
| `EquivariantLayer` | Trait: `forward(&features)`, `check_equivariance(&g, &features, tol)`. Layer that commutes with group action. |
| `SymmetryType` | Enum: Permutation, Translation{dim}, Rotation{dim}, Scale, Time, Gauge{fiber_dim}, GroupConv{order}. |
| `AgentFeatures` | Feature matrix (n_agents × feature_dim) with `n_agents()`, `feature_dim()`, `from_matrix()`, `norm()`, `scale()`. |
| `AgentGraph` | Adjacency structure with `new(n)`, `add_edge(i,j,w)`, `neighbors(i)`, `num_agents`, `normalized_laplacian()`, `ring(n)`. |
| `GeometricModelConfig` | Configuration: input_dim, hidden_dim, output_dim, num_layers, symmetries, use_spectral, use_spatial. |

---

### `permutation` — Permutation Equivariance

| Type | Description |
|---|---|
| `Permutation` | Group element: `new(perm)`, `identity(n)`, `from_seed(n, seed)`, `inverse()`, `compose(&other)`, `apply_vec(&v)`. |
| `Aggregation` | Enum: Sum, Mean, Max. |
| `DeepSetsLayer` | Permutation-equivariant layer: φ(individual) → aggregate → ρ(combined). Implements `EquivariantLayer`. |

---

### `translation` — Translation Equivariance

| Type | Description |
|---|---|
| `Translation` | Group element with shift vector. Implements `GroupAction`. |
| `TranslationEquivLayer` | Weight-sharing linear layer that ignores absolute position. |

---

### `rotation` — Rotation Equivariance

| Type | Description |
|---|---|
| `RotationAction` | Enum: Rot2{angle}, Rot3{matrix}. Implements `GroupAction`. SO(2) and SO(3) rotations. |
| `RotationInvariantFeatures` | Compute invariant features: distances, angles, areas. Configurable via flags. |
| `SONEquivariantLayer` | SO(n)-equivariant layer using scalar invariants + learnable projection. |

**`RotationAction` methods:**
- `rot2(angle)`, `rot3_from_axis_angle(&axis, angle)`
- `rotate_positions(&positions) → DMatrix<f64>`

---

### `scale` — Scale Equivariance

| Type | Description |
|---|---|
| `ScaleAction` | Group element with scale factor. Implements `GroupAction`. |
| `ScaleMode` | Enum: Normalize, LogScale, MultiScale{scales}. |
| `ScaleEquivLayer` | Scale-equivariant layer. Implements `EquivariantLayer`. |
| `MultiResolutionPool` | Aggregate features at multiple scales: `pool()`, `pool_concat()`. |

---

### `time_equiv` — Time Equivariance

| Type | Description |
|---|---|
| `TimeShift` | Group element with shift steps. Implements `GroupAction`. |
| `TemporalConvLayer` | Causal 1D convolution with learnable kernel: `convolve_causal(&signal)`, `forward(&time_series)`. |
| `TemporalRecurrentLayer` | Simple RNN: `forward(&sequence)`, `forward_reverse(&sequence)`. |

**Free functions:**
- `temporal_differences(&time_series, order) → DMatrix<f64>` — k-th order finite differences.
- `running_average(&time_series, window) → DMatrix<f64>` — Causal moving average.

---

### `spectral` — Spectral Filters

| Type | Description |
|---|---|
| `SpectralFilter` | Polynomial filter g(Λ) in the graph frequency domain. |
| `GraphFourier` | Static methods for graph Fourier transform. |

**`SpectralFilter`:**
- `new(order)`, `with_coefficients(coeffs)`
- `low_pass(order, cutoff)`, `high_pass(order, cutoff)` — Preset filters
- `apply_exact(&graph, &features)` — Full eigendecomposition
- `apply_chebyshev(&graph, &features)` — Chebyshev approximation (faster)
- `evaluate(lambda) → f64` — Evaluate filter at an eigenvalue

**`GraphFourier`:**
- `transform(&graph, &signal) → DVector<f64>` — Forward GFT
- `inverse_transform(&graph, &spectrum) → DVector<f64>` — Inverse GFT
- `frequencies(&graph) → DVector<f64>` — Laplacian eigenvalues
- `spectral_pool(&graph, &signal, keep_ratio) → DVector<f64>` — Low-pass pooling

---

### `spatial` — Spatial Message Passing

| Type | Description |
|---|---|
| `MessageAggregation` | Enum: Sum, Mean, Max. |
| `MessagePassingLayer` | MPNN: message → aggregate → update. `message(&h_i, &h_j)`, `aggregate(&messages)`, `update(&h, &m)`, `forward(&graph, &features)`. |
| `GraphConvLayer` | Graph convolution with optional attention: `new(in, out)`, `with_attention()`, `attention(&h_i, &h_j)`, `forward(&graph, &features)`. |
| `GNNScheme` | Multi-layer GNN: `new(&[dims], aggregation)`, `forward(&graph, &features)`, `readout_sum(&features)`, `readout_attention(&features)`. |

---

### `gauge` — Gauge Equivariance

| Type | Description |
|---|---|
| `GaugeTransformation` | Per-agent coordinate change: `new(gauges)`, `identity(n, d)`, `random_orthogonal(n, d, seed)`. Implements `GroupAction`. |
| `GaugeEquivLayer` | Layer that commutes with gauge transformations: `new(input_dim, output_dim)`, `forward(&features)`, `check_gauge_equivariance(&gauge, &features, tol)`. |

---

### `group_conv` — Group Convolution

| Type | Description |
|---|---|
| `FiniteGroup` | Group by multiplication table: `cyclic(n)`, `dihedral(n)`. Fields: `order`, `table`, `inv`. |
| `GroupConvLayer` | G-equivariant convolution: `(f * g)(x) = Σ f(y) g(y⁻¹x)`. `new(group, input_dim, output_dim)`, `forward(&features)`. |

---

### `agent` — Unified Agent Model

| Type | Description |
|---|---|
| `EquivariantAgentModel` | Full model composing all equivariant layers. `new(config)`, `forward(&graph, &features) → AgentFeatures`. |

---

### `universal` — Universal Approximation

| Type | Description |
|---|---|
| `UniversalApproximator` | Multi-layer Deep Sets with configurable width/depth. `new(input_dim, hidden_dim, output_dim, depth)`, `forward(&features)`, `empirical_error(&features, &target) → f64`. |

## How It Works

1. **GroupAction trait**: Every symmetry (permutation, translation, rotation, scale, time shift, gauge change) implements `GroupAction` with `act`, `inverse`, `compose`, `identity`.

2. **EquivariantLayer trait**: Each layer implements `forward` and gets `check_equivariance` for free — it verifies f(g·x) = g·f(x) numerically.

3. **Deep Sets** (permutation): f(X) = ρ(Σᵢ φ(xᵢ)) where φ is a per-element MLP and ρ is a post-aggregation MLP. This is equivariant by construction.

4. **Spectral filters**: Diagonalize the graph Laplacian L = UΛUᵀ, apply a learnable polynomial g(Λ) in the frequency domain: output = U g(Λ) Uᵀ X. Chebyshev approximation avoids the O(n³) eigendecomposition.

5. **Message passing**: For each node, compute messages from neighbors (concatenate + linear + ReLU), aggregate (sum/mean/max), update (concatenate with old features + linear + ReLU).

6. **Group convolution**: For finite groups, the convolution (f * g)(x) = Σ_{y∈G} f(y) g(y⁻¹x) is G-equivariant by construction (it's the group algebra product).

7. **Gauge equivariance**: Transform features by gauge matrices at each node, then apply the layer. Gauge-equivariant layers commute with all such transformations.

## The Math

### Equivariance

A function f is G-equivariant if f(g·x) = g·f(x) for all g ∈ G. This is a constraint on the function space — it restricts f to a subspace that respects the symmetry. The key insight: you get better generalization by *building in* equivariance rather than *learning* it.

### Deep Sets (Zaheer et al.)

Any continuous permutation-invariant function on sets can be decomposed as ρ(Σᵢ φ(xᵢ)) where φ and ρ are continuous functions. For equivariant functions: f(x₁,...,xₙ)ᵢ = ρ(xᵢ, Σⱼ φ(xⱼ)). This is the universal approximation theorem for sets.

### Graph Fourier Transform

The graph Laplacian L = D − A has eigendecomposition L = UΛUᵀ. The columns of U are the "Fourier modes" and Λ contains the "frequencies". The GFT of signal x is x̂ = Uᵀx.

### Chebyshev Filters

Instead of eigendecomposition, approximate the spectral filter using Chebyshev polynomials: g(L) ≈ Σₖ cₖ Tₖ(L̃) where L̃ is the scaled Laplacian. The recurrence Tₖ(x) = 2xTₖ₋₁(x) − Tₖ₋₂(x) gives O(K|E|) complexity.

### Group Convolution

On a finite group G, convolution (f * g)(x) = Σ_{y∈G} f(y)g(y⁻¹x) is equivariant: (f * g)(hx) = h(f * g)(x). This generalizes circular convolution (G = Z_n) and planar convolution (G = R²).

## License

MIT
