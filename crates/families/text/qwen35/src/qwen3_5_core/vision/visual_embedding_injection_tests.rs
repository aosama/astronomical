use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

use super::qwen3_5_inject_visual_embeddings;

const IMAGE_PAD_TOKEN_ID: u32 = 9;
const HIDDEN_WIDTH: i32 = 2;

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(256 * 1024 * 1024, 256 * 1024 * 1024)
            .expect("the splice test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

/// Three text positions, the middle one an image pad, in `dtype`.
fn text_embeddings(runtime: &MlxRuntime, dtype: MlxDtype) -> MlxArray {
    let float32_text = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 3, HIDDEN_WIDTH])
        .expect("the text embeddings should build");
    runtime
        .astype(&float32_text, dtype)
        .expect("the text embeddings should cast")
}

fn visual_embeddings(runtime: &MlxRuntime, dtype: MlxDtype, width: i32) -> MlxArray {
    let values: Vec<f32> = (0..width).map(|column| -0.5 - column as f32).collect();
    let float32_visual = runtime
        .array_from_f32(&values, &[1, width])
        .expect("the visual embeddings should build");
    runtime
        .astype(&float32_visual, dtype)
        .expect("the visual embeddings should cast")
}

fn spliced_values(runtime: &MlxRuntime, spliced: &MlxArray) -> Vec<f32> {
    runtime
        .astype(spliced, MlxDtype::Float32)
        .expect("the splice should read as float32")
        .to_vec_f32()
        .expect("the splice should evaluate")
}

#[test]
#[serial]
fn should_widen_bfloat16_visual_rows_into_float32_text_embeddings() {
    let runtime = test_runtime();
    let (spliced, spliced_row_count) = qwen3_5_inject_visual_embeddings(
        &runtime,
        &text_embeddings(&runtime, MlxDtype::Float32),
        &[1, IMAGE_PAD_TOKEN_ID, 2],
        &visual_embeddings(&runtime, MlxDtype::BFloat16, HIDDEN_WIDTH),
        0,
        IMAGE_PAD_TOKEN_ID,
    )
    .expect("a BF16 vision tower splices into a float32 language model");

    assert_eq!(spliced_row_count, 1);
    assert_eq!(spliced.dtype(), MlxDtype::Float32);
    assert_eq!(
        spliced_values(&runtime, &spliced),
        vec![1.0, 2.0, -0.5, -1.5, 5.0, 6.0]
    );
}

#[test]
#[serial]
fn should_splice_matching_bfloat16_embeddings_as_they_are() {
    let runtime = test_runtime();
    let (spliced, _spliced_row_count) = qwen3_5_inject_visual_embeddings(
        &runtime,
        &text_embeddings(&runtime, MlxDtype::BFloat16),
        &[1, IMAGE_PAD_TOKEN_ID, 2],
        &visual_embeddings(&runtime, MlxDtype::BFloat16, HIDDEN_WIDTH),
        0,
        IMAGE_PAD_TOKEN_ID,
    )
    .expect("matching dtypes splice");

    assert_eq!(spliced.dtype(), MlxDtype::BFloat16);
}

#[test]
#[serial]
fn should_refuse_a_wider_visual_dtype_than_the_text_and_a_mismatched_width() {
    let runtime = test_runtime();
    for (text_dtype, visual_dtype, visual_width) in [
        (MlxDtype::BFloat16, MlxDtype::Float32, HIDDEN_WIDTH),
        (MlxDtype::Float32, MlxDtype::BFloat16, HIDDEN_WIDTH + 1),
    ] {
        assert!(
            qwen3_5_inject_visual_embeddings(
                &runtime,
                &text_embeddings(&runtime, text_dtype),
                &[1, IMAGE_PAD_TOKEN_ID, 2],
                &visual_embeddings(&runtime, visual_dtype, visual_width),
                0,
                IMAGE_PAD_TOKEN_ID,
            )
            .is_err(),
            "{visual_dtype:?} rows of width {visual_width} into {text_dtype:?} text must be refused"
        );
    }
}
