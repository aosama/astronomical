//! Compile-time graph transformations and compiled graph execution.

pub(crate) mod attention_output_gate;
pub(crate) mod elementwise_graphs;
pub(crate) mod graph;
pub(crate) mod multi_output_graph;
pub(crate) mod sparse_shared_expert_combination;
pub(crate) mod swiglu;
pub(crate) mod transforms;
pub(crate) mod verify_window;
pub(crate) mod vision_rope;
