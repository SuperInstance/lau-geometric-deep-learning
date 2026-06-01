# lau-geometric-deep-learning

**Geometric Deep Learning framework — 5 symmetries applied to agent systems.** Permutation, translation, rotation, scale, and time equivariance with spectral and spatial graph filters, gauge-equivariant layers, and group convolutions.

132 tests · MIT license · `nalgebra` + `serde`

---

## What This Does

This crate implements the Bronstein et al. Geometric Deep Learning blueprint in Rust, targeting **agent systems on graphs**. An agent system is a set of agents with features, connected by a graph. The core question: *how do you build neural network layers that respect the symmetries of your data?*

The five symmetries implemented:

1. **Permutation equivariance** — agent order doesn't matter (Deep Sets, Set Transformers)
2. **Translation equivariance** — shift-invariant features via relative coordinates and circular convolution
3. **Rotation equivariance** — SO(2)/SO(3) equivariant features via invariant distances and angles
4. **Scale equivariance** — multi-resolution features via normalization and log-scale
5. **Time equivariance** — temporal shift equivariance via causal convolution and recurrent layers

Plus: spectral graph filters (Chebyshev + exact eigendecomposition), spatial message passing (MPNN, GAT-like attention), gauge-equivariant layers with parallel transport, group convolution on finite groups (cyclic Z_n, dihedral D_n, symmetric S_n), and universal approximation theory.

---

## Key Idea

**Equivariance**: A function f is equivariant to a symmetry group G if f(g·x) = g·f(x) for all g ∈ G. If you permute the agents, the output permutes the same way. If you rotate the configuration, the output rotates too.

This isn't just mathematical elegance — it's a **strong inductive bias** that dramatically reduces the function space the network needs to search, giving better generalization with less data.

---

## Install

```toml
[dependencies]
lau-geometric-deep-learning = "0.1.0"
```

Dependencies: `nalgebra = "0.33"` (with serde + rand), `serde = "1"`, `serde_json = "1"`.

---

## Quick Start

```rust
use lau_geometric_deep_learning::prelude::*;
use nalgebra::DMatrix;

fn main() {
    // Build a graph
    let mut graph = AgentGraph::new(5);
    graph.add_edge(0, 1, 1.0);
    graph.add_edge(1, 2, 1.0);
    graph.add_edge(2, 3, 1.0);
    graph.add_edge(3, 4, 1.0);

    // Create features (5 agents × 8 features each)
    let features = AgentFeatures::random(5, 8);

    // Build a permutation-equivariant model
    let model = EquivariantModelBuilder::new()
        .input_dim(8)
        .hidden_dim(16)
        .output_dim(4)
        .num_layers(2)
        .num_agents(5)
        .add_symmetry(SymmetryType::Permutation)
        .add_symmetry(SymmetryType::Scale)
        .use_spatial(true)
        .build();

    // Forward pass
    let output = model.forward(&graph, &features);
    println!("Output: {} agents × {} features", output.n_agents(), output.feature_dim());
}
```

---

## API Reference

### Core Types (`core`)

| Type | Description |
|------|-------------|
| `AgentGraph` | Graph structure: adjacency matrix, degree matrix, Laplacian, normalized Laplacian |
| `AgentFeatures` | Feature matrix (n_agents × feature_dim) with permute, translate, scale operations |
| `GeometricModelConfig` | Configuration: input/hidden/output dims, layers, symmetries, spectral/spatial flags |
| `SymmetryType` | Enum: `Permutation`, `Translation{dim}`, `Rotation{dim}`, `Scale`, `Time`, `Gauge{fiber_dim}`, `GroupConv{order}` |
| `GroupAction` | Trait: `act`, `inverse`, `compose`, `identity` |
| `EquivariantLayer` | Trait: `forward`, `check_equivariance` |

**`AgentGraph`** methods:
- `new(n)`, `fully_connected(n)`, `ring(n)`, `knn(positions, k)`
- `add_edge(i, j, w)`, `neighbors(i)`, `num_edges()`, `is_connected()`
- `degree_matrix()`, `laplacian()`, `normalized_laplacian()`

### Permutation Equivariance (`permutation`)

| Type | Description |
|------|-------------|
| `Permutation` | Group element: `new(perm)`, `identity(n)`, `from_seed(n, seed)`, `inverse()`, `compose()`, `to_matrix()` |
| `DeepSetsLayer` | φ-ρ architecture with Sum/Mean/Max aggregation |
| `SetTransformerLayer` | Attention-based permutation equivariant layer |
| `Aggregation` | Enum: `Sum`, `Mean`, `Max` |

| Function | Description |
|----------|-------------|
| `check_permutation_equivariance(f, features, perm, tol)` | Verify equivariance of arbitrary function |

### Translation Equivariance (`translation`)

| Type | Description |
|------|-------------|
| `Translation` | Shift vector group element |
| `TranslationEquivLayer` | Uses relative coordinates (centroid-subtracted) or pairwise differences |
| `CircularConvLayer` | 1D circular convolution for ring-structured agents |

### Rotation Equivariance (`rotation`)

| Type | Description |
|------|-------------|
| `RotationAction` | Enum: `Rot2{angle}`, `Rot3{matrix}` — SO(2)/SO(3) group element |
| `RotationInvariantFeatures` | Computes distances, angles, cos(angle) — rotation-invariant by construction |
| `SONEquivariantLayer` | SO(n)-equivariant layer using scalar invariants |

### Scale Equivariance (`scale`)

| Type | Description |
|------|-------------|
| `ScaleAction` | Scalar multiplication group element |
| `ScaleEquivLayer` | Normalize / LogScale / MultiScale modes |
| `ScaleMode` | `Normalize` (L2), `LogScale` (log|x|), `MultiScale{scales}` |
| `MultiResolutionPool` | Pool features at multiple scales, concatenate |

### Time Equivariance (`time_equiv`)

| Type | Description |
|------|-------------|
| `TimeShift` | Cyclic temporal shift group element |
| `TemporalConvLayer` | Causal 1D convolution |
| `TemporalRecurrentLayer` | Recurrent layer with forward and reverse modes |

| Function | Description |
|----------|-------------|
| `temporal_differences(ts, order)` | Compute nth-order temporal differences |
| `running_average(ts, window)` | Causal running average smoothing |

### Spectral Filters (`spectral`)

| Type | Description |
|------|-------------|
| `SpectralFilter` | Polynomial filter in graph frequency domain |
| `GraphFourier` | Static: `transform`, `inverse_transform`, `frequencies`, `spectral_pool` |

`SpectralFilter` methods:
- `new(order)`, `low_pass(order, cutoff)`, `high_pass(order, cutoff)`
- `apply_exact(graph, features)` — via full eigendecomposition: U·g(Λ)·Uᵀ·X
- `apply_chebyshev(graph, features)` — via Chebyshev polynomial approximation (faster)
- `evaluate(λ)` — evaluate polynomial filter at eigenvalue

### Spatial Filters (`spatial`)

| Type | Description |
|------|-------------|
| `MessagePassingLayer` | MPNN with message/aggregate/update functions |
| `MessageAggregation` | `Sum`, `Mean`, `Max` |
| `GraphConvLayer` | Graph convolution with optional GAT-like attention |
| `GNNScheme` | Multi-layer GNN stack with readout (sum/attention) |

### Gauge Equivariance (`gauge`)

| Type | Description |
|------|-------------|
| `GaugeTransformation` | Per-agent orthogonal gauge matrices; `identity(n, fd)`, `random_orthogonal(n, fd, seed)` |
| `GaugeEquivLayer` | Gauge-equivariant layer using parallel transport on graph |
| `Connection` | Parallel transport matrices; `trivial(n, fd)`, `holonomy(cycle)` |

### Group Convolution (`group_conv`)

| Type | Description |
|------|-------------|
| `FiniteGroup` | Group via multiplication table; `cyclic(n)`, `dihedral(n)`, `symmetric(n)` |
| `GroupConvLayer` | G-equivariant convolution: (f∗k)(x) = Σ_y f(y)·k(y⁻¹x) |
| `LiftingLayer` | Lifts scalar features to regular representation |

### Unified Model (`agent`)

| Type | Description |
|------|-------------|
| `EquivariantAgentModel` | Full pipeline combining all equivariant layers |
| `EquivariantModelBuilder` | Builder pattern for configuring models |

`EquivariantAgentModel` methods:
- `new(config)` — construct from config
- `forward(graph, features)` — full forward pass
- `extract_features(graph, features)` — returns DMatrix
- `verify_equivariance(graph, features, tol)` — check permutation equivariance

### Universal Approximation (`universal`)

| Type | Description |
|------|-------------|
| `UniversalApproximator` | Deep Sets stack with configurable depth |
| `IrrepType` | `Scalar`, `Vector{dim}` — irreducible representation types |
| `ClebschGordanNet` | Irrep composition network |

---

## How It Works

### Architecture

```
AgentGraph ──→ EquivariantModelBuilder ──→ EquivariantAgentModel
                    │                          │
                    ├─ SymmetryType ────────┐  │
                    │  (Permutation)        │  ├─ permutation_layers: DeepSetsLayer
                    │  (Translation)        │  ├─ message_passing_layers: MPNN
                    │  (Rotation)           │  ├─ spectral_filters: Chebyshev/exact
                    │  (Scale)              │  ├─ scale_layer: ScaleEquivLayer
                    │  (Gauge)              │  ├─ rotation_features: invariants
                    │  (GroupConv)          │  └─ gauge_layer: GaugeEquivLayer
                    └─ use_spectral         │
                       use_spatial          │
```

Each equivariant layer satisfies: f(g·x) = g·f(x) for its symmetry group G.

### Key Algorithms

**Deep Sets (permutation)**: φ transforms individual features, ρ transforms the aggregate: f(X) = ρ(Σᵢ φ(xᵢ))

**Message Passing (spatial)**: For each node i: m_i = Σ_{j∈N(i)} MSG(h_i, h_j), h_i' = UPDATE(h_i, m_i)

**Spectral Filtering**: Diagonalize Laplacian L = UΛUᵀ, apply polynomial g(Λ), reconstruct: U·g(Λ)·Uᵀ·X

**Chebyshev Approximation**: Avoid eigendecomposition by computing T_k(L̃)·X recursively, scaling L to [-1,1]

**Group Convolution**: On finite group G with |G|=n: (f∗k)(x) = Σ_{y∈G} f(y)·k(y⁻¹x), automatically G-equivariant

**Gauge Equivariance**: Features live in a fiber bundle. Gauge transformation applies per-node orthogonal matrices. Parallel transport moves features between nodes via connection matrices.

---

## The Math

### Equivariance

A function f: X → Y is **G-equivariant** if f(g·x) = g·f(x) for all g ∈ G.

Special case: f is **G-invariant** if f(g·x) = f(x) (Y has trivial G-action).

### Graph Laplacian

L = D - A (combinatorial), L_norm = I - D^{-1/2}AD^{-1/2} (normalized)

Eigenvalues 0 = λ₁ ≤ λ₂ ≤ ... ≤ λₙ encode graph structure. λ₂ (algebraic connectivity) measures how connected the graph is.

### Graph Fourier Transform

For L = UΛUᵀ:

- Forward: f̂ = Uᵀf (decompose into graph frequencies)
- Inverse: f = Uf̂ (reconstruct)
- Filtering: f_filtered = U·g(Λ)·Uᵀ·f

### Chebyshev Polynomials

T₀(x) = 1, T₁(x) = x, Tₖ(x) = 2x·Tₖ₋₁(x) - Tₖ₋₂(x)

Any polynomial filter g(λ) can be approximated by Σₖ cₖTₖ(λ̃) where λ̃ is the scaled eigenvalue.

### Deep Sets Universality

Any continuous permutation-invariant function on sets can be decomposed as ρ(Σᵢ φ(xᵢ)) for suitable φ, ρ (Zaheer et al., 2017). This crate implements this as the `UniversalApproximator`.

### Group Convolution on Finite Groups

For finite group G with elements {g₀, ..., g_{n-1}}:

(f ∗ k)(x) = Σᵢ f(gᵢ) · k(gᵢ⁻¹x)

This is automatically G-equivariant: (f∗k)(hx) = h·(f∗k)(x).

### Gauge Theory on Graphs

A **connection** on a graph assigns a linear map (transport matrix) to each edge. **Holonomy** around a cycle is the product of transport matrices. Trivial connection → identity holonomy (flat geometry). Non-trivial connection → curvature.

---

## License

MIT
