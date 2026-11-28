//! An `astro_ssd_prompt_cache` block written from host copies (the path the
//! background writer takes) loads back through MLX with every element equal,
//! bit for bit, to the state that was published, including a strided view
//! like the engine's key-value block slices.

use std::collections::HashMap;
use std::time::Duration;

use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::MlxRuntime;

use super::astro_ssd_prompt_cache_disk_store_support::{
    astro_ssd_prompt_cache_block_key_for_seed, load_published_block_state,
    open_astro_ssd_prompt_cache_disk_store, runtime_with_shared_limits, synthetic_kv_block_tensors,
    synthetic_recurrent_snapshot_tensors,
};

/// Room for one production-sized block with no quota pressure.
const STORE_LIMIT_BYTES: u64 = 10 * 1_073_741_824;
const HOST_WRITTEN_RESTORE_TEST_TIMEOUT: Duration = Duration::from_secs(115);

#[tokio::test]
async fn should_restore_every_published_element_bit_for_bit_from_host_written_files() {
    tokio::time::timeout(HOST_WRITTEN_RESTORE_TEST_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let runtime = runtime_with_shared_limits();
        let store_directory = tempfile::tempdir().expect("a store directory");
        let store = open_astro_ssd_prompt_cache_disk_store(&store_directory, STORE_LIMIT_BYTES)
            .expect("the store opens");
        let mut sequence_state_tensors =
            patterned_tensors(&runtime, &synthetic_kv_block_tensors(&runtime));
        let boundary_state_tensors =
            patterned_tensors(&runtime, &synthetic_recurrent_snapshot_tensors(&runtime));
        replace_first_tensor_with_patterned_view(&runtime, &mut sequence_state_tensors);
        let block_key = astro_ssd_prompt_cache_block_key_for_seed(0);
        let publication = store.publish_block(
            &runtime,
            &block_key,
            None,
            &sequence_state_tensors,
            &boundary_state_tensors,
        );
        assert!(
            publication.is_ok(),
            "the patterned block publishes: {publication:?}"
        );
        let loaded_block_state = load_published_block_state(&runtime, &store, &block_key);
        assert_elements_match_bit_for_bit(
            &loaded_block_state.sequence_state_tensors,
            &sequence_state_tensors,
        );
        assert_elements_match_bit_for_bit(
            &loaded_block_state.boundary_state_tensors,
            &boundary_state_tensors,
        );
    })
    .await
    .expect("the bit-for-bit restore test should finish within 115 seconds");
}

/// Values far from zero and different along every run of 4,093 elements, so
/// a payload laid out at the wrong offset or in the wrong order cannot
/// compare equal.
fn patterned_values(element_count: usize) -> Vec<f32> {
    (0..element_count)
        .map(|element_index| (element_index % 4_093) as f32 * 0.5 - 1_000.0)
        .collect()
}

fn patterned_tensor(runtime: &MlxRuntime, shape: &[i32], dtype: MlxDtype) -> MlxArray {
    let element_count = shape.iter().map(|dimension| *dimension as usize).product();
    let float_tensor = runtime
        .array_from_f32(&patterned_values(element_count), shape)
        .expect("the patterned values should form an array");
    runtime
        .astype(&float_tensor, dtype)
        .expect("the patterned array should cast to the state dtype")
}

fn patterned_tensors(
    runtime: &MlxRuntime,
    shaped_tensors: &HashMap<String, MlxArray>,
) -> HashMap<String, MlxArray> {
    let mut patterned_tensors = HashMap::with_capacity(shaped_tensors.len());
    for (tensor_name, shaped_tensor) in shaped_tensors {
        patterned_tensors.insert(
            tensor_name.clone(),
            patterned_tensor(runtime, &shaped_tensor.shape(), shaped_tensor.dtype()),
        );
    }
    patterned_tensors
}

/// Swaps one tensor for a view one element in from a wider backing array, so
/// its storage is strided: the host copy must read it in logical order.
fn replace_first_tensor_with_patterned_view(
    runtime: &MlxRuntime,
    tensors: &mut HashMap<String, MlxArray>,
) {
    let tensor_name = tensors
        .keys()
        .min()
        .expect("the block should hold a sequence tensor")
        .clone();
    let expected_shape = tensors[&tensor_name].shape();
    let mut backing_shape = expected_shape.clone();
    let final_axis = backing_shape.len() - 1;
    backing_shape[final_axis] += 1;
    let backing = patterned_tensor(runtime, &backing_shape, tensors[&tensor_name].dtype());
    let mut slice_starts = vec![0_i32; backing_shape.len()];
    slice_starts[final_axis] = 1;
    let strided_view = runtime
        .slice(
            &backing,
            &slice_starts,
            &backing_shape,
            &vec![1_i32; backing_shape.len()],
        )
        .expect("the strided view should slice");
    assert_eq!(strided_view.shape(), expected_shape);
    tensors.insert(tensor_name, strided_view);
}

fn element_bits(tensor: &MlxArray) -> Vec<u32> {
    match tensor.dtype() {
        MlxDtype::BFloat16 => widened_bits(tensor.to_vec_bf16_bits().expect("bfloat16 copies")),
        MlxDtype::Float16 => widened_bits(tensor.to_vec_f16_bits().expect("float16 copies")),
        MlxDtype::Float32 => tensor
            .to_vec_f32()
            .expect("float32 copies")
            .into_iter()
            .map(f32::to_bits)
            .collect(),
        other_dtype => panic!("no decoder state is stored as {other_dtype:?}"),
    }
}

fn widened_bits(sixteen_bit_elements: Vec<u16>) -> Vec<u32> {
    sixteen_bit_elements.into_iter().map(u32::from).collect()
}

fn assert_elements_match_bit_for_bit(
    loaded_tensors: &HashMap<String, MlxArray>,
    published_tensors: &HashMap<String, MlxArray>,
) {
    assert_eq!(loaded_tensors.len(), published_tensors.len());
    for (tensor_name, published_tensor) in published_tensors {
        let loaded_tensor = &loaded_tensors[tensor_name];
        assert_eq!(
            loaded_tensor.shape(),
            published_tensor.shape(),
            "{tensor_name}"
        );
        assert!(
            element_bits(loaded_tensor) == element_bits(published_tensor),
            "{tensor_name} should restore bit for bit"
        );
    }
}
