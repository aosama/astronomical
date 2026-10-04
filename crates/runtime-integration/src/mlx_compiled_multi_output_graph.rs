use crate::{
    MlxArray, MlxRuntime, MlxRuntimeError,
    mlx_compiled_graph::{MlxCompiledGraph, MlxGraphBuilder},
};

const APPLY_COMPILED_MULTI_OUTPUT_GRAPH_OPERATION: &str = "apply a compiled multi-output MLX graph";

/// Owns one compiled MLX graph that produces an ordered vector of outputs.
///
/// Model-side compiled composites that need several outputs at once — the
/// multi-token-prediction verification window returns successor states beside
/// its logits — construct one of these with their own graph builder and apply
/// it through [`MlxRuntime::apply_compiled_multi_output_graph`]. Output order
/// is the builder's write order and never changes after compilation.
#[derive(Debug)]
pub struct MlxCompiledMultiOutputGraph {
    compiled_graph: MlxCompiledGraph,
}

impl MlxCompiledMultiOutputGraph {
    /// Compiles the given graph builder into a replayable multi-output graph.
    /// See [`crate::mlx_compiled_graph::MlxCompiledGraph::new`] for the
    /// shapeless tradeoff.
    pub fn new(
        graph_builder: MlxGraphBuilder,
        compile_operation: &'static str,
        shapeless: bool,
    ) -> Result<Self, MlxRuntimeError> {
        Ok(Self {
            compiled_graph: MlxCompiledGraph::new(graph_builder, compile_operation, shapeless)?,
        })
    }
}

impl MlxRuntime {
    /// Applies a compiled multi-output graph and returns its ordered outputs.
    pub fn apply_compiled_multi_output_graph(
        &self,
        compiled_graph: &MlxCompiledMultiOutputGraph,
        graph_inputs: &[&MlxArray],
    ) -> Result<Vec<MlxArray>, MlxRuntimeError> {
        compiled_graph
            .compiled_graph
            .apply_multi(graph_inputs, APPLY_COMPILED_MULTI_OUTPUT_GRAPH_OPERATION)
    }
}
