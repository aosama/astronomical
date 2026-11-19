//! MLX's gradients through every operation Laya's forward pass uses, checked
//! against the loss's own change along a direction (a directional
//! derivative), which needs no hand-written backward pass to compare with.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, MlxDtype};
use serial_test::serial;

use super::{TrainableLoss, loss_gradients};
use astronomical_model_primitives::synthetic_weights_test_support::{
    SYNTHETIC_QUANTIZATION_BITS, SYNTHETIC_QUANTIZATION_GROUP_SIZE, initialize_test_runtime,
    shakespeare_values,
};

const ROW_COUNT: i32 = 2;
const SEQUENCE_LENGTH: i32 = 6;
const HIDDEN_WIDTH: i32 = 64;
const HEAD_COUNT: i32 = 2;
const ADAPTER_RANK: i32 = 4;

/// A one-layer miniature of Laya: LayerNorm, an 8-bit fused projection with
/// an add-on, rotary embedding, masked attention, exact GELU and a read at
/// two marker positions, squared and summed into one number.
struct MiniatureLayerLoss {
    input_states: MlxArray,
    norm_weight: MlxArray,
    packed_projection: MlxArray,
    projection_scales: MlxArray,
    projection_biases: MlxArray,
    padding_mask: MlxArray,
    marker_positions: MlxArray,
}

impl MiniatureLayerLoss {
    fn new(test_runtime: &MlxRuntime) -> Self {
        let runtime: &MlxBindingsContext = test_runtime;
        let dense_projection =
            shakespeare_values(test_runtime, &[3 * HIDDEN_WIDTH, HIDDEN_WIDTH], 900, 0.0);
        let (packed_projection, projection_scales, projection_biases) = runtime
            .quantize_affine(
                &dense_projection,
                SYNTHETIC_QUANTIZATION_GROUP_SIZE,
                SYNTHETIC_QUANTIZATION_BITS,
            )
            .unwrap();
        // The second row's last two positions are padding.
        let mut mask_values = vec![0.0_f32; (ROW_COUNT * SEQUENCE_LENGTH) as usize];
        mask_values[10] = -1e9;
        mask_values[11] = -1e9;
        Self {
            input_states: shakespeare_values(
                test_runtime,
                &[ROW_COUNT, SEQUENCE_LENGTH, HIDDEN_WIDTH],
                0,
                0.0,
            ),
            norm_weight: shakespeare_values(test_runtime, &[HIDDEN_WIDTH], 400, 1.0),
            packed_projection,
            projection_scales,
            projection_biases,
            padding_mask: runtime
                .array_from_f32(&mask_values, &[ROW_COUNT, 1, 1, SEQUENCE_LENGTH])
                .unwrap(),
            marker_positions: runtime
                .array_from_i32(&[1, 4, 0, 3], &[ROW_COUNT, 2, 1])
                .unwrap(),
        }
    }

    fn attention(
        &self,
        context: &MlxBindingsContext,
        fused_projection: &MlxArray,
    ) -> Result<MlxArray, MlxCError> {
        let head_width = HIDDEN_WIDTH / HEAD_COUNT;
        let split = context.reshape(
            fused_projection,
            &[ROW_COUNT, SEQUENCE_LENGTH, 3, HEAD_COUNT, head_width],
        )?;
        let reordered = context.transpose_axes(&split, &[0, 3, 2, 1, 4])?;
        let mut projections = Vec::with_capacity(3);
        for projection_index in 0..3 {
            let one_projection = context.slice(
                &reordered,
                &[0, 0, projection_index, 0, 0],
                &[
                    ROW_COUNT,
                    HEAD_COUNT,
                    projection_index + 1,
                    SEQUENCE_LENGTH,
                    head_width,
                ],
                &[1, 1, 1, 1, 1],
            )?;
            projections.push(context.squeeze_axis(&one_projection, 2)?);
        }
        let queries = context.rope(&projections[0], head_width, 10_000.0, 0)?;
        let keys = context.rope(&projections[1], head_width, 10_000.0, 0)?;
        let attended = context.masked_scaled_dot_product_attention(
            &queries,
            &keys,
            &projections[2],
            (head_width as f32).powf(-0.5),
            &self.padding_mask,
        )?;
        let sequence_ordered = context.transpose_axes(&attended, &[0, 2, 1, 3])?;
        context.reshape(
            &sequence_ordered,
            &[ROW_COUNT, SEQUENCE_LENGTH, HIDDEN_WIDTH],
        )
    }
}

impl TrainableLoss for MiniatureLayerLoss {
    fn build_loss(
        &self,
        context: &MlxBindingsContext,
        trainable_arrays: &[MlxArray],
    ) -> Result<MlxArray, MlxCError> {
        let (adapter_down, adapter_up) = (&trainable_arrays[0], &trainable_arrays[1]);
        let zero_bias = context.zeros(&[HIDDEN_WIDTH], MlxDtype::Float32)?;
        let normalized =
            context.layer_norm(&self.input_states, &self.norm_weight, &zero_bias, 1e-5)?;
        let base_projection = context.quantized_matmul_affine(
            &normalized,
            &self.packed_projection,
            &self.projection_scales,
            &self.projection_biases,
            true,
            SYNTHETIC_QUANTIZATION_GROUP_SIZE,
            SYNTHETIC_QUANTIZATION_BITS,
        )?;
        let narrowed =
            context.matmul(&normalized, &context.transpose_axes(adapter_down, &[1, 0])?)?;
        let adapter_projection =
            context.matmul(&narrowed, &context.transpose_axes(adapter_up, &[1, 0])?)?;
        let fused_projection = context.add(&base_projection, &adapter_projection)?;
        let attended = self.attention(context, &fused_projection)?;
        let activated = context.gelu(&attended)?;
        let marker_states = context.take_along_axis(&activated, &self.marker_positions, 1)?;
        context.sum(&context.square(&marker_states)?, false)
    }
}

/// A loss that never reads its second trainable array.
struct FirstArrayOnlyLoss;

impl TrainableLoss for FirstArrayOnlyLoss {
    fn build_loss(
        &self,
        context: &MlxBindingsContext,
        trainable_arrays: &[MlxArray],
    ) -> Result<MlxArray, MlxCError> {
        context.sum(&context.square(&trainable_arrays[0])?, false)
    }
}

/// A loss differentiated with respect to an 8-bit packed weight, which MLX
/// cannot do: only the activations of a quantized product have a gradient.
struct PackedWeightLoss {
    activations: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
}

impl TrainableLoss for PackedWeightLoss {
    fn build_loss(
        &self,
        context: &MlxBindingsContext,
        trainable_arrays: &[MlxArray],
    ) -> Result<MlxArray, MlxCError> {
        let projected = context.quantized_matmul_affine(
            &self.activations,
            &trainable_arrays[0],
            &self.scales,
            &self.biases,
            true,
            SYNTHETIC_QUANTIZATION_GROUP_SIZE,
            SYNTHETIC_QUANTIZATION_BITS,
        )?;
        context.sum(&projected, false)
    }
}

fn scalar(array: &MlxArray) -> f64 {
    f64::from(array.to_vec_f32().unwrap()[0])
}

fn dot(left: &MlxArray, right: &MlxArray) -> f64 {
    let left_values = left.to_vec_f32().unwrap();
    let right_values = right.to_vec_f32().unwrap();
    let mut dot_product = 0.0_f64;
    for (left_value, right_value) in left_values.iter().zip(&right_values) {
        dot_product += f64::from(*left_value) * f64::from(*right_value);
    }
    dot_product
}

fn stepped(
    context: &MlxBindingsContext,
    start: &MlxArray,
    direction: &MlxArray,
    step: f32,
) -> MlxArray {
    context
        .add(start, &context.multiply_scalar(direction, step).unwrap())
        .unwrap()
}

#[test]
#[serial]
fn should_match_the_loss_change_along_a_direction_through_every_laya_operation() {
    let runtime = initialize_test_runtime();
    let adapter_down = shakespeare_values(&runtime, &[ADAPTER_RANK, HIDDEN_WIDTH], 1700, 0.0);
    let adapter_up = shakespeare_values(&runtime, &[3 * HIDDEN_WIDTH, ADAPTER_RANK], 2100, 0.0);
    let down_direction = shakespeare_values(&runtime, &[ADAPTER_RANK, HIDDEN_WIDTH], 3100, 0.0);
    let up_direction = shakespeare_values(&runtime, &[3 * HIDDEN_WIDTH, ADAPTER_RANK], 3500, 0.0);

    let differentiated = loss_gradients(
        MiniatureLayerLoss::new(&runtime),
        &[adapter_down.retain().unwrap(), adapter_up.retain().unwrap()],
    )
    .unwrap();
    let predicted_change = dot(&differentiated.gradients[0], &down_direction)
        + dot(&differentiated.gradients[1], &up_direction);

    let reference_loss = MiniatureLayerLoss::new(&runtime);
    // The loss is about 1.6 and float32 rounds it near 1e-7 relative, plus
    // TF32 in the add-on's products; a step of 0.01 leaves the difference
    // quotient 10-30% noisy, while 0.1 keeps it within 1% and the loss still
    // close enough to linear along the direction.
    let step = 1e-1_f32;
    let mut stepped_losses = Vec::with_capacity(2);
    for signed_step in [step, -step] {
        let stepped_arrays = [
            stepped(&runtime, &adapter_down, &down_direction, signed_step),
            stepped(&runtime, &adapter_up, &up_direction, signed_step),
        ];
        stepped_losses.push(scalar(
            &reference_loss
                .build_loss(&runtime, &stepped_arrays)
                .unwrap(),
        ));
    }
    let measured_change = (stepped_losses[0] - stepped_losses[1]) / (2.0 * f64::from(step));

    assert!(
        predicted_change.abs() > 1e-3,
        "the direction should change the loss measurably, got {predicted_change}"
    );
    let relative_gap = (predicted_change - measured_change).abs() / measured_change.abs();
    assert!(
        relative_gap < 2e-2,
        "gradient predicts {predicted_change}, the loss changed by {measured_change}"
    );
}

#[test]
#[serial]
fn should_give_a_zero_gradient_to_an_array_the_loss_never_reads() {
    let runtime = initialize_test_runtime();
    let read_array = shakespeare_values(&runtime, &[3, 4], 0, 0.5);
    let unread_array = shakespeare_values(&runtime, &[5], 100, 0.5);

    let differentiated = loss_gradients(FirstArrayOnlyLoss, &[read_array, unread_array]).unwrap();

    assert_eq!(differentiated.gradients.len(), 2);
    assert_eq!(differentiated.gradients[1].shape(), vec![5]);
    assert!(
        differentiated.gradients[1]
            .to_vec_f32()
            .unwrap()
            .iter()
            .all(|gradient_value| *gradient_value == 0.0)
    );
}

#[test]
#[serial]
fn should_report_an_error_when_a_graph_has_no_gradient_for_an_input() {
    let runtime = initialize_test_runtime();
    let dense_weight = shakespeare_values(&runtime, &[64, 64], 0, 0.0);
    let (packed_weight, scales, biases) = runtime
        .quantize_affine(
            &dense_weight,
            SYNTHETIC_QUANTIZATION_GROUP_SIZE,
            SYNTHETIC_QUANTIZATION_BITS,
        )
        .unwrap();
    let packed_weight_loss = PackedWeightLoss {
        activations: shakespeare_values(&runtime, &[2, 64], 500, 0.0),
        scales,
        biases,
    };

    let differentiated = loss_gradients(packed_weight_loss, &[packed_weight]);

    assert!(differentiated.is_err());
}
