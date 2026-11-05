//! Contracts that need Core ML or MLX but no Neural Engine and no real model, so
//! they are not ignored and run in the integration lane.

mod document_compile_contracts;
mod mlx_hand_off_contracts;
