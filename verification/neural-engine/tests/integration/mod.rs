//! Contracts that need Core ML but no Neural Engine, no MLX runtime, and no real
//! model, so they are not ignored and run in the integration lane.

mod document_compile_contracts;
