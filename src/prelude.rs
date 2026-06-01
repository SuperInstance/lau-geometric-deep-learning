//! Prelude: re-export commonly used types.

pub use crate::core::{
    AgentFeatures, AgentGraph, EquivariantLayer, GeometricModelConfig, GroupAction, SymmetryType,
};
pub use crate::permutation::{Aggregation, DeepSetsLayer, Permutation, SetTransformerLayer};
pub use crate::translation::{CircularConvLayer, Translation, TranslationEquivLayer};
pub use crate::rotation::{RotationAction, RotationInvariantFeatures, SONEquivariantLayer};
pub use crate::scale::{MultiResolutionPool, ScaleAction, ScaleEquivLayer, ScaleMode};
pub use crate::time_equiv::{TemporalConvLayer, TemporalRecurrentLayer, TimeShift};
pub use crate::spectral::{GraphFourier, SpectralFilter};
pub use crate::spatial::{GNNScheme, GraphConvLayer, MessageAggregation, MessagePassingLayer};
pub use crate::gauge::{Connection, GaugeEquivLayer, GaugeTransformation};
pub use crate::group_conv::{FiniteGroup, GroupConvLayer};
pub use crate::agent::{EquivariantAgentModel, EquivariantModelBuilder};
pub use crate::universal::{ApproximationCapacity, IrrepType, SteerableNetwork, UniversalApproximator};
