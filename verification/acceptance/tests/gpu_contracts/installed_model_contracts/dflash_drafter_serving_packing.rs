//! A drafter packed ahead of time (`astronomical quantize drafter`) serves
//! bit for bit the weights the engine makes from the bf16 checkpoint when
//! it loads: every linear's 4-bit codes, scales and biases, and every
//! codebook. That equality is what lets a package ship the packed drafter
//! at about a third of the download (issue #2595).
//!
//! The configuration names a model directory whose drafter folder holds a
//! bf16 DFlash drafter.

use std::fs::File;

use mlx_bridge::MlxArray;
use model_serving::speculative_decoding::dflash_drafter::drafter_linear::DrafterLinear;
use model_serving::speculative_decoding::dflash_drafter::{
    DFlashDrafterConfiguration, DFlashDrafterWeights, bind_drafter_weights,
    drafter_tensors_at_serving_packing,
};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::file_serial;

use super::dflash_drafter_load_memory::bundled_drafter_directory;

#[test]
#[ignore = "loads a real drafter on the GPU; run with --configuration naming a model with a bf16 drafter"]
#[file_serial]
fn should_serve_a_drafter_packed_ahead_of_time_bit_for_bit_as_the_engine_packs_it_at_load() {
    let drafter_directory = bundled_drafter_directory();
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(16 * 1024 * 1024 * 1024, 256 * 1024 * 1024)
            .expect("memory limits are valid"),
    )
    .expect("the MLX runtime initializes");
    let configuration = drafter_configuration(&drafter_directory);
    let packed_directory = tempfile::tempdir().expect("a scratch folder");
    let packed_weights_path = packed_directory.path().join("model.safetensors");
    let packed_tensors =
        drafter_tensors_at_serving_packing(&runtime, &drafter_directory.join("model.safetensors"))
            .expect("the drafter packs");
    let named_tensors: Vec<(&str, &MlxArray)> = packed_tensors
        .iter()
        .map(|(tensor_name, tensor)| (tensor_name.as_str(), tensor))
        .collect();
    runtime
        .save_safetensors(
            File::create(&packed_weights_path).expect("the packed file opens"),
            &named_tensors,
            &[("format", "mlx")],
        )
        .expect("the packed drafter writes");

    let weights_packed_at_load = served_drafter_weights(
        &runtime,
        &drafter_directory.join("model.safetensors"),
        &configuration,
    );
    let weights_packed_ahead =
        served_drafter_weights(&runtime, &packed_weights_path, &configuration);

    let served_arrays_at_load = served_arrays(&weights_packed_at_load);
    let served_arrays_ahead = served_arrays(&weights_packed_ahead);
    assert_eq!(served_arrays_at_load.len(), served_arrays_ahead.len());
    let mut differing_arrays = Vec::new();
    for ((array_name, array_at_load), (_, array_ahead)) in
        served_arrays_at_load.iter().zip(&served_arrays_ahead)
    {
        if !arrays_are_equal(&runtime, array_at_load, array_ahead) {
            differing_arrays.push(array_name.clone());
        }
    }
    println!(
        "[dflash-drafter-serving-packing] compared_arrays={} differing={differing_arrays:?}",
        served_arrays_at_load.len()
    );
    assert!(
        differing_arrays.is_empty(),
        "every served array must match: {differing_arrays:?}"
    );
}

fn drafter_configuration(drafter_directory: &std::path::Path) -> DFlashDrafterConfiguration {
    let config_document: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(drafter_directory.join("config.json"))
            .expect("the drafter's config reads"),
    )
    .expect("the drafter's config parses");
    DFlashDrafterConfiguration::from_config_document(&config_document)
        .expect("the drafter's configuration is accepted")
}

/// The drafter's weights as the engine serves them on a GPU without the
/// register-ordered kernel: every linear in MLX's 4-bit packing.
fn served_drafter_weights(
    runtime: &MlxRuntime,
    weights_path: &std::path::Path,
    configuration: &DFlashDrafterConfiguration,
) -> DFlashDrafterWeights {
    let checkpoint = runtime
        .load_safetensors(File::open(weights_path).expect("the weights open"), None)
        .expect("the checkpoint maps");
    let mut drafter_weights = bind_drafter_weights(runtime, &checkpoint, configuration)
        .expect("every drafter tensor binds");
    drop(checkpoint);
    drafter_weights
        .quantize_linears(runtime, false)
        .expect("the linears pack");
    drafter_weights
}

/// Every array the served drafter multiplies or gathers, named, in a fixed
/// order: each linear's codes, scales and biases, then the codebooks.
fn served_arrays(drafter_weights: &DFlashDrafterWeights) -> Vec<(String, MlxArray)> {
    let mut named_linears: Vec<(String, &DrafterLinear)> =
        vec![("fc".to_owned(), &drafter_weights.tap_fusion)];
    for (layer_index, layer) in drafter_weights.layers.iter().enumerate() {
        for (projection_name, projection) in [
            ("q_proj", &layer.query_projection),
            ("k_proj", &layer.key_projection),
            ("v_proj", &layer.value_projection),
            ("o_proj", &layer.output_projection),
            ("gate_proj", &layer.gate_projection),
            ("up_proj", &layer.up_projection),
            ("down_proj", &layer.down_projection),
        ] {
            named_linears.push((
                format!("layers.{layer_index}.{projection_name}"),
                projection,
            ));
        }
        if let Some(convolutions) = &layer.convolutions {
            named_linears.push((
                format!("layers.{layer_index}.attention_conv"),
                &convolutions.attention.kernel_projection,
            ));
            named_linears.push((
                format!("layers.{layer_index}.mlp_conv"),
                &convolutions.mlp.kernel_projection,
            ));
        }
    }
    let mut arrays = Vec::new();
    for (linear_name, linear) in named_linears {
        let DrafterLinear::Quantized {
            packed_weight,
            scales,
            biases,
        } = linear
        else {
            panic!("{linear_name} is served in MLX's 4-bit packing");
        };
        for (part_name, part) in [
            ("weight", packed_weight),
            ("scales", scales),
            ("biases", biases),
        ] {
            arrays.push((
                format!("{linear_name}.{part_name}"),
                part.retain().expect("the array handle retains"),
            ));
        }
    }
    if let Some(candidate_selector) = &drafter_weights.candidate_selector {
        for (codebook_name, codebook) in [
            (
                "predecessor_codebook",
                &candidate_selector.predecessor_codebook,
            ),
            ("successor_codebook", &candidate_selector.successor_codebook),
        ] {
            arrays.push((
                codebook_name.to_owned(),
                codebook.retain().expect("the array handle retains"),
            ));
        }
    }
    arrays
}

fn arrays_are_equal(runtime: &MlxRuntime, first_array: &MlxArray, second_array: &MlxArray) -> bool {
    let equality = runtime
        .array_equal(first_array, second_array, false)
        .expect("the arrays compare");
    runtime
        .evaluate_arrays(&[&equality])
        .expect("the comparison runs");
    equality.item_bool().expect("the comparison reads back")
}
