//! Metal source of the sorted-expert 4-bit affine matmul. The Metal code is
//! in the `metal` folder beside this file, as Metal Shading Language (C++)
//! files; this module brings each one in at compile time and documents what
//! it computes, its inputs and its template integers.
//!
//! The arithmetic is MLX's own `affine_gather_qmm_rhs_nax` (MLX 0.32.5,
//! `mlx/backend/metal/kernels/quantized_nax.h`), as TensorFold re-expresses it
//! bit for bit (`zig/kernels/metal/prefill/qmm_nax.metal` and
//! `zig/kernels/metal/nax.h`, Apache-2.0, Copyright 2026 TensorFold
//! contributors): each 64-deep step dequantizes a weight block to bf16 exactly
//! as MLX does (`scale * q + bias`, the high nibble as `(scale / 16) * (q & 0xf0)`)
//! and feeds 16x32x16 matrix-unit operations in the same order. Every stored
//! output element therefore sees the same operands in the same order as in
//! MLX, and comes out with the same bits.
//!
//! What differs is how the work is laid out (`SortedExpertLayout`; the
//! measurements chose 128 columns, staged rows, 96-row tiles):
//! - A row tile holds up to 96 rows of one expert, so most experts have their
//!   weights dequantized once per column block; a popular expert's tiles run
//!   on several threadgroups at once, as in MLX's schedule. A 16-row fragment
//!   with no live row skips the matrix units.
//! - A step's token rows can be staged once in threadgroup memory for every
//!   simdgroup; MLX's simdgroups each read them from device memory.
//! - The token-row variant reads sorted row r straight from token
//!   `order[r] / TOPK`'s row, so the engine's sorted copy of the token rows
//!   (a 64 MiB take for a 2,048-token chunk) is never made. Against that copy
//!   plus MLX's gather_qmm for gate and up it measured 13% less GPU time,
//!   and a served 35B-A3B prompt chunk read 1.0% faster with the same logits.
//! - Each expert's first row is found by one thread per expert (a binary
//!   search over the sorted ids), not by MLX's one-threadgroup scan.
//!
//! Requirements the source does not check: the M5 matrix units (MLX compiles
//! the source on first use, so serving must prove it first), and inputs that
//! start on 16-byte boundaries, because rows, packed bytes and fragments are
//! read with vector loads. Whole MLX arrays and row slices of them do.

/// The M5 matrix-unit fragments (TensorFold's `nax.h`, M5 path) and the
/// dequantization.
pub const SORTED_EXPERT_MATMUL_HEADER: &str = include_str!("metal/sorted_expert_matmul.h");

/// Inputs: `ids` [M] the sorted expert of every row, `row_count` [1] int32 (M).
/// Output: `first_rows` [E] int32, the first row whose expert is not below e.
/// One thread per expert.
pub const EXPERT_FIRST_ROWS_SOURCE: &str = include_str!("metal/expert_first_rows.metal");

/// Inputs: `X` [M, K] bf16 rows sorted by expert, MLX's packed `Wq`
/// [E, N, K / 8] uint32 with its bf16 `S` scales and `B` biases [E, N, K / 64],
/// `first_rows` [E] int32, `row_count` [1] int32 (M). Output: `Y` [M, N] bf16.
///
/// `TILE_SCHEDULE` 1: one threadgroup per BM-row tile of an expert (MLX's
/// schedule), so a popular expert's rows run on several threadgroups at
/// once. 0: one threadgroup per expert (grid y = E), taking its rows BM at a
/// time. Either way an expert of up to BM rows has its weights dequantized
/// once per column block.
///
/// Template integers: `N`, `K`, `E`, `BM` rows a pass (64 or 96: two
/// simdgroups down, BM / 32 16-row fragments each), `WN` simdgroups across
/// the columns (a pass is BM rows x 32 WN columns, 64 WN threads), and
/// `STAGE_ROWS`: 1 stages a step's token rows in threadgroup memory once for
/// every simdgroup, 0 has each simdgroup read them from device memory as MLX
/// does.
pub const SORTED_EXPERT_MATMUL_SOURCE: &str = include_str!("metal/sorted_expert_matmul.metal");

/// The sorted rows are X's own rows: sorted row r is X's row r.
pub const SORTED_ROWS_PREAMBLE: &str = include_str!("metal/sorted_rows_preamble.metal");

/// The sorted rows are read through the sort: X holds one row per token, and
/// sorted row r is token `order[r] / TOPK`'s row, so no sorted copy of the
/// token rows is ever made. Only the staged layouts read through it.
pub const TOKEN_ROWS_PREAMBLE: &str = include_str!("metal/token_rows_preamble.metal");
