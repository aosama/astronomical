//! Which inputs the row-exact tree attention kernel takes. Anything it does
//! not take falls back to MLX's attention, so a refusal must be a clean
//! `false`, never an error halfway through a verify. The kernel's own
//! arithmetic is held to the one-row step and to float32 attention by
//! `cargo measure-row-exact-tree-attention`.

use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::file_serial;

use super::gated_delta_tree::TreeRowChains;
use super::row_exact_tree_attention::{
    RowExactTreeAttentionInputs, fits_attention_shape, row_exact_tree_attention_accepts,
};

const HEAD_SIZE: i32 = 256;
const KEY_VALUE_HEAD_COUNT: i32 = 2;
const CONTEXT_LENGTH: i32 = 100;

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the tree attention test memory limits are valid"),
    )
    .expect("the direct MLX runtime initializes")
}

/// The shapes one acceptance check is made from.
struct Shapes {
    query_heads: i32,
    rows: usize,
    head_size: i32,
    slab_capacity: i32,
    query_dtype: MlxDtype,
}

impl Shapes {
    /// The 35B-A3B target's 4-row round, its slab filled exactly.
    fn fitting() -> Self {
        Self {
            query_heads: 16,
            rows: 4,
            head_size: HEAD_SIZE,
            slab_capacity: CONTEXT_LENGTH + 4,
            query_dtype: MlxDtype::BFloat16,
        }
    }

    fn accepted(&self, runtime: &MlxRuntime) -> bool {
        let mut parent_rows = vec![None];
        for row in 1..self.rows {
            parent_rows.push(Some(row - 1));
        }
        let row_chains = TreeRowChains::from_parent_rows(&parent_rows)
            .upload(runtime)
            .expect("the chains upload");
        let queries = zeros(
            runtime,
            &[1, self.query_heads, self.rows as i32, self.head_size],
            self.query_dtype,
        );
        let slab_shape = [1, KEY_VALUE_HEAD_COUNT, self.slab_capacity, self.head_size];
        let key_slab = zeros(runtime, &slab_shape, MlxDtype::BFloat16);
        let value_slab = zeros(runtime, &slab_shape, MlxDtype::BFloat16);
        row_exact_tree_attention_accepts(&RowExactTreeAttentionInputs {
            queries: &queries,
            key_slab: &key_slab,
            value_slab: &value_slab,
            context_length: CONTEXT_LENGTH,
            row_chains: &row_chains,
            scale: 1.0 / 16.0,
        })
    }
}

fn zeros(runtime: &MlxRuntime, shape: &[i32], dtype: MlxDtype) -> MlxArray {
    runtime.zeros(shape, dtype).expect("zeros build")
}

#[test]
#[file_serial]
fn should_accept_the_mixture_of_experts_round_with_its_slab_filled_exactly() {
    assert!(Shapes::fitting().accepted(&test_runtime()));
}

/// Three rows in a chain pad their chains to 4 entries; the slab holds only
/// the 3 rows after the context, and that is all the kernel reads.
#[test]
#[file_serial]
fn should_accept_a_three_row_chain_whose_padded_chain_passes_the_slab_end() {
    let shapes = Shapes {
        rows: 3,
        slab_capacity: CONTEXT_LENGTH + 3,
        ..Shapes::fitting()
    };
    assert!(shapes.accepted(&test_runtime()));
}

#[test]
#[file_serial]
fn should_accept_a_single_row() {
    let shapes = Shapes {
        rows: 1,
        slab_capacity: CONTEXT_LENGTH + 1,
        ..Shapes::fitting()
    };
    assert!(shapes.accepted(&test_runtime()));
}

/// A dense target's tree: 16 rows of 6 query heads a key-value head fill 6
/// matrix tiles.
#[test]
#[file_serial]
fn should_accept_a_dense_targets_sixteen_row_tree() {
    let shapes = Shapes {
        query_heads: 6 * KEY_VALUE_HEAD_COUNT,
        rows: 16,
        slab_capacity: CONTEXT_LENGTH + 16,
        ..Shapes::fitting()
    };
    assert!(shapes.accepted(&test_runtime()));
}

/// Edge: 16 rows of 8 query heads fill the 8 tiles a threadgroup holds.
#[test]
#[file_serial]
fn should_accept_sixteen_rows_that_fill_every_matrix_tile() {
    let shapes = Shapes {
        rows: 16,
        slab_capacity: CONTEXT_LENGTH + 16,
        ..Shapes::fitting()
    };
    assert!(shapes.accepted(&test_runtime()));
}

#[test]
#[file_serial]
fn should_refuse_more_than_sixteen_rows() {
    let shapes = Shapes {
        query_heads: 6 * KEY_VALUE_HEAD_COUNT,
        rows: 17,
        slab_capacity: CONTEXT_LENGTH + 17,
        ..Shapes::fitting()
    };
    assert!(!shapes.accepted(&test_runtime()));
}

#[test]
#[file_serial]
fn should_refuse_a_slab_without_room_for_every_row() {
    let shapes = Shapes {
        slab_capacity: CONTEXT_LENGTH + 3,
        ..Shapes::fitting()
    };
    assert!(!shapes.accepted(&test_runtime()));
}

#[test]
#[file_serial]
fn should_refuse_a_query_group_other_than_six_or_eight() {
    let shapes = Shapes {
        query_heads: 8,
        ..Shapes::fitting()
    };
    assert!(!shapes.accepted(&test_runtime()));
}

#[test]
#[file_serial]
fn should_refuse_float32_queries() {
    let shapes = Shapes {
        query_dtype: MlxDtype::Float32,
        ..Shapes::fitting()
    };
    assert!(!shapes.accepted(&test_runtime()));
}

#[test]
#[file_serial]
fn should_refuse_a_head_size_other_than_256() {
    let shapes = Shapes {
        head_size: 128,
        ..Shapes::fitting()
    };
    assert!(!shapes.accepted(&test_runtime()));
}

#[test]
fn should_fit_the_two_attention_shapes_the_kernel_is_proven_on() {
    // The 35B-A3B (16 query heads on 2) and the dense 27B (24 on 4).
    assert!(fits_attention_shape(256, 16, 2));
    assert!(fits_attention_shape(256, 24, 4));
}

#[test]
fn should_not_fit_another_head_size_group_or_an_uneven_grouping() {
    assert!(!fits_attention_shape(128, 16, 2));
    assert!(!fits_attention_shape(256, 8, 2));
    // Edge: 10 query heads do not divide over 4 key-value heads, and a
    // model with no key-value head has no group at all.
    assert!(!fits_attention_shape(256, 10, 4));
    assert!(!fits_attention_shape(256, 16, 0));
}
