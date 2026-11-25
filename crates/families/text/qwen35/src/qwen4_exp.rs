//! Qwen3.8-Flash-Next (`qwen4_exp`): the pieces it adds to the Qwen 3.5
//! decoder stack (issue #2533). Everything it shares with Qwen 3.5 stays in
//! the Qwen 3.5 modules; these modules hold only what differs.

#[cfg(feature = "direct-mlx")]
pub mod centered_norm;
#[cfg(feature = "direct-mlx")]
pub mod growing_sequence;
#[cfg(feature = "direct-mlx")]
pub mod hyper_connection;
#[cfg(feature = "direct-mlx")]
pub mod layer_weights;
pub mod ngram_hashing;
#[cfg(feature = "direct-mlx")]
pub mod ngram_layer;
#[cfg(feature = "direct-mlx")]
pub mod ngram_table;
pub mod ngram_table_geometry;
#[cfg(feature = "direct-mlx")]
pub mod request_state;
#[cfg(feature = "direct-mlx")]
pub mod sparse_attention;
