//! Prompt-cache restore reconstruction strategy probe.
//!
//! Production (persistent_state_kv_restore.rs) assembles the restored
//! full-attention KV with one `concatenate` per layer, O(restored tokens).
//! This probe archives the comparison that decided that design: the replaced
//! per-block path preallocated a final-length destination and, for each block,
//! wrote it with `slice_update` then materialized. MLX has no in-place slice
//! write, so every `slice_update` recopied the whole destination and every
//! block forced a GPU synchronization; the path was super-linear in restored
//! tokens.
//!
//! The probe measures the replaced pattern against the production concat on
//! identical data and scales total tokens at the real 2048-token cache block
//! size so the super-linear cost stays visible next to the single-pass concat:
//!   A "current-per-block" = per-block `slice_update` + per-block evaluate
//!   B "batched-eval"      = per-block `slice_update` + one final evaluate
//!   C "single-concat"     = one `concatenate` over all blocks + one evaluate
//! Every strategy is asserted to assemble bit-identical restored state.

use std::time::{Duration, Instant};

use astronomical_mlx_c_rust::{MlxArray, MlxDtype};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use tokio::time::timeout;

const KV_HEAD_COUNT: i32 = 16;
const KV_HEAD_DIMENSION: i32 = 128;
const BLOCK_TOKENS: i32 = 2048;
const TOTAL_TOKEN_POINTS: &[i32] = &[8192, 16384, 32768];
const PROBE_ROUNDS: usize = 11;

#[tokio::test]
#[ignore = "measures prompt-cache restore reconstruction strategies on the GPU with interleaved medians"]
async fn should_probe_prompt_cache_restore_reconstruction_costs() {
    timeout(Duration::from_secs(115), probe_restore_reconstruction())
        .await
        .expect("the restore reconstruction probe must finish within 115 seconds");
}

async fn probe_restore_reconstruction() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(16 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024)
            .expect("the probe memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");

    for &total_tokens in TOTAL_TOKEN_POINTS {
        let block_count = (total_tokens / BLOCK_TOKENS) as usize;
        let block_shape = [1, KV_HEAD_COUNT, BLOCK_TOKENS, KV_HEAD_DIMENSION];
        let block_element_count = (BLOCK_TOKENS * KV_HEAD_COUNT * KV_HEAD_DIMENSION) as usize;
        let blocks: Vec<MlxArray> = (0..block_count)
            .map(|block_index| {
                bf16_array(
                    &runtime,
                    &block_values(block_element_count, block_index),
                    &block_shape,
                )
            })
            .collect();

        let destination_shape: [i32; 4] = [1, KV_HEAD_COUNT, total_tokens, KV_HEAD_DIMENSION];
        let reference = runtime
            .concatenate_axis(&blocks.iter().collect::<Vec<_>>(), 2)
            .expect("the reference concat should build");
        runtime
            .evaluate_arrays(&[&reference])
            .expect("the reference should evaluate");

        let runtime_ref: &MlxRuntime = &runtime;
        let blocks_ref: &[MlxArray] = &blocks;
        let destination_shape_ref: &[i32] = &destination_shape;
        let probes: Vec<(&str, Box<dyn Fn() -> MlxArray>)> = vec![
            (
                "current-per-block",
                Box::new(move || {
                    per_block_slice_update(runtime_ref, blocks_ref, destination_shape_ref, true)
                }),
            ),
            (
                "batched-eval",
                Box::new(move || {
                    per_block_slice_update(runtime_ref, blocks_ref, destination_shape_ref, false)
                }),
            ),
            (
                "single-concat",
                Box::new(move || single_concat(runtime_ref, blocks_ref)),
            ),
        ];

        for (_, probe) in &probes {
            let warmup_output = probe();
            runtime
                .evaluate_arrays(&[&warmup_output])
                .expect("the warmup probe should evaluate");
        }
        runtime
            .synchronize_gpu_stream()
            .expect("the warmup probes should drain");

        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); probes.len()];
        for _round in 0..PROBE_ROUNDS {
            for (probe_index, (_, probe)) in probes.iter().enumerate() {
                let started_at = Instant::now();
                probe();
                samples[probe_index].push(started_at.elapsed().as_secs_f64() * 1000.0);
            }
        }
        runtime
            .synchronize_gpu_stream()
            .expect("the probes should drain");

        let mut medians = Vec::with_capacity(probes.len());
        for (probe_name, _) in &probes {
            let probe_samples = &mut samples[medians.len()];
            probe_samples.sort_by(|left, right| left.partial_cmp(right).expect("finite times"));
            let median = probe_samples[probe_samples.len() / 2];
            medians.push(median);
            eprintln!(
                "[restore-reconstruction-probe] tokens={total_tokens} blocks={block_count} {probe_name} median {median:.3} ms samples={}",
                probe_samples.len(),
            );
        }
        eprintln!(
            "[restore-reconstruction-probe] tokens={total_tokens} current/batched = {:.2}  current/concat = {:.2} (a ratio above 1.0 means current is slower)",
            medians[0] / medians[1],
            medians[0] / medians[2],
        );

        let reference_values = runtime
            .astype(&reference, MlxDtype::Float32)
            .expect("the reference should cast")
            .to_vec_f32()
            .expect("the reference should evaluate");
        for (probe_name, probe) in &probes {
            let probe_values = runtime
                .astype(&probe(), MlxDtype::Float32)
                .expect("the probe output should cast")
                .to_vec_f32()
                .expect("the probe output should evaluate");
            let maximum_difference = reference_values
                .iter()
                .zip(probe_values.iter())
                .map(|(reference_value, probe_value)| (reference_value - probe_value).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                maximum_difference == 0.0,
                "{probe_name} must assemble the identical restored state: max diff {maximum_difference}"
            );
        }
    }
}

fn per_block_slice_update(
    runtime: &MlxRuntime,
    blocks: &[MlxArray],
    destination_shape: &[i32],
    evaluate_per_block: bool,
) -> MlxArray {
    let mut destination = runtime
        .zeros(destination_shape, MlxDtype::BFloat16)
        .expect("the restore destination should allocate");
    let mut offset_tokens = 0_i32;
    for block in blocks {
        let block_token_count = block.shape()[2];
        let mut starts = vec![0_i32; destination_shape.len()];
        starts[2] = offset_tokens;
        let mut stops = destination_shape.to_vec();
        stops[2] = offset_tokens + block_token_count;
        let strides = vec![1_i32; destination_shape.len()];
        destination = runtime
            .slice_update(&destination, block, &starts, &stops, &strides)
            .expect("the block should slice into the destination");
        offset_tokens += block_token_count;
        if evaluate_per_block {
            runtime
                .evaluate_arrays(&[&destination])
                .expect("the destination should materialize");
        }
    }
    if !evaluate_per_block {
        runtime
            .evaluate_arrays(&[&destination])
            .expect("the destination should materialize");
    }
    destination
}

fn single_concat(runtime: &MlxRuntime, blocks: &[MlxArray]) -> MlxArray {
    let assembled = runtime
        .concatenate_axis(&blocks.iter().collect::<Vec<_>>(), 2)
        .expect("the blocks should concatenate");
    runtime
        .evaluate_arrays(&[&assembled])
        .expect("the concatenated state should materialize");
    assembled
}

fn bf16_array(runtime: &MlxRuntime, values: &[f32], shape: &[i32]) -> MlxArray {
    runtime
        .astype(
            &runtime
                .array_from_f32(values, shape)
                .expect("the bfloat16 source should be valid"),
            MlxDtype::BFloat16,
        )
        .expect("the array should cast to bfloat16")
}

fn block_values(element_count: usize, block_index: usize) -> Vec<f32> {
    (0..element_count)
        .map(|value_index| ((value_index % 29) as f32 - 14.0) * 0.05 + (block_index as f32) * 0.25)
        .collect()
}
