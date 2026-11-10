//! The same sorted rows must give the same bits whichever page holds their experts.
//!
//! The paged engine multiplies a layer's experts from different pages: a streamed routed
//! union, a warm-table subset, or the complete table. MLX `gather_qmm` picks its tiled or
//! per-row kernel from the assignment count per expert of the tensor it receives, and the
//! two kernels round differently. These contracts multiply one quantized layer both ways,
//! with the order the engines declare, so an MLX upgrade that moves that rule fails here
//! instead of silently changing seeded answers (issue #1679).

use astronomical_model_serving::{
    ExpertAssignmentOrder, PerformanceAttribution, StackedExpertProjection,
    expert_assignment_order, gather_expert_projection,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_bridge::{MlxArray, MlxDtype};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

/// The Qwen 3.5 35B-A3B layer width in experts, the geometry the defect showed on.
const LAYER_EXPERT_COUNT: i32 = 256;
/// A routed union small enough that a short chunk reaches four assignments per page expert.
const PAGE_EXPERT_COUNT: i32 = 64;
const INPUT_WIDTH: i32 = 256;
const OUTPUT_WIDTH: i32 = 64;
const QUANTIZATION_GROUP_SIZE: i32 = 64;
const QUANTIZATION_BITS: i32 = 4;
/// A 48-token chunk at eight experts per token: under four per layer expert.
const SHORT_CHUNK_ASSIGNMENT_COUNT: i32 = 384;
/// Four assignments per layer expert: the smallest chunk the tiled kernel may take.
const TILED_FLOOR_ASSIGNMENT_COUNT: i32 = 1_024;

struct AffineExpertStack {
    packed_weights: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
}

impl AffineExpertStack {
    fn projection(&self) -> StackedExpertProjection<'_> {
        StackedExpertProjection::Affine {
            packed_weights: &self.packed_weights,
            scales: &self.scales,
            biases: &self.biases,
            group_size: QUANTIZATION_GROUP_SIZE,
            bits: QUANTIZATION_BITS,
        }
    }
}

/// One quantized layer, and the page of every fourth expert taken from it.
struct LayerAndPage {
    complete_table: AffineExpertStack,
    routed_page: AffineExpertStack,
}

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("page independence test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

/// The page holds the experts whose global ID is four times its slot.
fn global_expert_id_for_page_slot(page_slot: i32) -> i32 {
    page_slot * (LAYER_EXPERT_COUNT / PAGE_EXPERT_COUNT)
}

/// Spread values, so a changed accumulation order shows in the rounded result.
fn varied_value(value_index: i32, stride: i32) -> f32 {
    (((value_index * stride) % 997) as f32 - 498.0) / 1_024.0
}

fn layer_and_page(runtime: &MlxRuntime) -> LayerAndPage {
    let weight_value_count = LAYER_EXPERT_COUNT * OUTPUT_WIDTH * INPUT_WIDTH;
    let mut weight_values = Vec::with_capacity(weight_value_count as usize);
    for value_index in 0..weight_value_count {
        weight_values.push(varied_value(value_index, 37));
    }
    let float32_layer_weights = runtime
        .array_from_f32(
            &weight_values,
            &[LAYER_EXPERT_COUNT, OUTPUT_WIDTH, INPUT_WIDTH],
        )
        .expect("the layer weights should be valid");
    let layer_weights = runtime
        .astype(&float32_layer_weights, MlxDtype::BFloat16)
        .expect("the layer weights should cast to bfloat16");
    let (packed_weights, scales, biases) = runtime
        .quantize_affine(&layer_weights, QUANTIZATION_GROUP_SIZE, QUANTIZATION_BITS)
        .expect("the layer weights should quantize");
    let mut page_expert_ids = Vec::with_capacity(PAGE_EXPERT_COUNT as usize);
    for page_slot in 0..PAGE_EXPERT_COUNT {
        page_expert_ids.push(global_expert_id_for_page_slot(page_slot) as u32);
    }
    let page_expert_id_array = runtime
        .array_from_u32(&page_expert_ids, &[PAGE_EXPERT_COUNT])
        .expect("the page expert ids should be valid");
    let routed_page = AffineExpertStack {
        packed_weights: runtime
            .take_axis(&packed_weights, &page_expert_id_array, 0)
            .expect("the page should copy packed rows"),
        scales: runtime
            .take_axis(&scales, &page_expert_id_array, 0)
            .expect("the page should copy scales"),
        biases: runtime
            .take_axis(&biases, &page_expert_id_array, 0)
            .expect("the page should copy biases"),
    };
    LayerAndPage {
        complete_table: AffineExpertStack {
            packed_weights,
            scales,
            biases,
        },
        routed_page,
    }
}

/// Sorted assignments over the page's experts: slot IDs for the page, and the same
/// experts' global IDs for the complete table.
fn sorted_page_slots_and_global_ids(assignment_count: i32) -> (Vec<u32>, Vec<u32>) {
    let mut page_slots = Vec::with_capacity(assignment_count as usize);
    let mut global_expert_ids = Vec::with_capacity(assignment_count as usize);
    for assignment_index in 0..assignment_count {
        let page_slot = assignment_index * PAGE_EXPERT_COUNT / assignment_count;
        page_slots.push(page_slot as u32);
        global_expert_ids.push(global_expert_id_for_page_slot(page_slot) as u32);
    }
    (page_slots, global_expert_ids)
}

fn sorted_activation_rows(runtime: &MlxRuntime, assignment_count: i32) -> MlxArray {
    let activation_value_count = assignment_count * INPUT_WIDTH;
    let mut activation_values = Vec::with_capacity(activation_value_count as usize);
    for value_index in 0..activation_value_count {
        activation_values.push(varied_value(value_index, 53));
    }
    let float32_activation_rows = runtime
        .array_from_f32(&activation_values, &[assignment_count, 1, INPUT_WIDTH])
        .expect("the activation rows should be valid");
    runtime
        .astype(&float32_activation_rows, MlxDtype::BFloat16)
        .expect("the activation rows should cast to bfloat16")
}

fn projected_values(
    runtime: &MlxRuntime,
    activation_rows: &MlxArray,
    expert_stack: &AffineExpertStack,
    expert_ids: &[u32],
    assignment_order: ExpertAssignmentOrder,
) -> Vec<f32> {
    let assignment_count = i32::try_from(expert_ids.len()).expect("the count fits i32");
    let expert_id_array = runtime
        .array_from_u32(expert_ids, &[assignment_count])
        .expect("the expert ids should be valid");
    let projected_rows = gather_expert_projection(
        runtime,
        activation_rows,
        expert_stack.projection(),
        &expert_id_array,
        assignment_order,
        &mut PerformanceAttribution::disabled(),
    )
    .expect("the gathered projection should build");
    runtime
        .astype(&projected_rows, MlxDtype::Float32)
        .expect("the projection should cast for the host")
        .to_vec_f32()
        .expect("the projection should evaluate")
}

/// Projects one chunk through the complete table and through the page, in that order.
fn table_and_page_values(
    assignment_count: i32,
    assignment_order: ExpertAssignmentOrder,
) -> (Vec<f32>, Vec<f32>) {
    let runtime = test_runtime();
    let layer_and_page = layer_and_page(&runtime);
    let (page_slots, global_expert_ids) = sorted_page_slots_and_global_ids(assignment_count);
    let activation_rows = sorted_activation_rows(&runtime, assignment_count);
    let table_values = projected_values(
        &runtime,
        &activation_rows,
        &layer_and_page.complete_table,
        &global_expert_ids,
        assignment_order,
    );
    let page_values = projected_values(
        &runtime,
        &activation_rows,
        &layer_and_page.routed_page,
        &page_slots,
        assignment_order,
    );
    (table_values, page_values)
}

fn engine_declared_order(assignment_count: i32) -> ExpertAssignmentOrder {
    expert_assignment_order(
        true,
        usize::try_from(assignment_count).expect("the count is positive"),
        usize::try_from(LAYER_EXPERT_COUNT).expect("the count is positive"),
    )
}

#[tokio::test]
async fn should_give_identical_bits_from_a_routed_page_and_the_complete_table_for_a_short_chunk() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let (table_values, page_values) = table_and_page_values(
        SHORT_CHUNK_ASSIGNMENT_COUNT,
        engine_declared_order(SHORT_CHUNK_ASSIGNMENT_COUNT),
    );
    assert_eq!(
        table_values, page_values,
        "a short chunk must not take a different kernel for a smaller page"
    );
}

#[tokio::test]
async fn should_give_identical_bits_from_a_routed_page_and_the_complete_table_at_the_tiled_floor() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let assignment_order = engine_declared_order(TILED_FLOOR_ASSIGNMENT_COUNT);
    assert_eq!(assignment_order, ExpertAssignmentOrder::SortedByExpert);
    let (table_values, page_values) =
        table_and_page_values(TILED_FLOOR_ASSIGNMENT_COUNT, assignment_order);
    assert_eq!(
        table_values, page_values,
        "at four assignments per layer expert every page takes the tiled kernel"
    );
}

#[tokio::test]
async fn should_show_that_claiming_sorted_ids_on_a_short_chunk_lets_the_page_change_the_bits() {
    // The control for the two contracts above: the claim production used to make. If
    // MLX stops choosing its kernel by page size, this fails, and the policy floor in
    // `assignment_sort_policy` can be revisited.
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let (table_values, page_values) = table_and_page_values(
        SHORT_CHUNK_ASSIGNMENT_COUNT,
        ExpertAssignmentOrder::SortedByExpert,
    );
    assert_ne!(
        table_values, page_values,
        "MLX no longer picks its quantized gather kernel by page size"
    );
}
