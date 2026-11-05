//! Weight values and how they are stored: half-precision conversion, row-major
//! blocks, and the compressed forms a model document can keep.

pub mod affine_quantized;
pub mod half_precision;
pub mod row_major_block;
pub mod shared_lookup_tables;
