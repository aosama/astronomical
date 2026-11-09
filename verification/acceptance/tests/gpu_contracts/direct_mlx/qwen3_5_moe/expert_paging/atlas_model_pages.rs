//! An Expert Atlas model must hand the expert math exactly the bytes its
//! stock source does, through the production plan builder and page loader.
//!
//! Run through `cargo xtask test-run --configuration FILE` whose
//! `model_directory` names an Atlas model and whose `reference_artifact_path`
//! names the stock checkpoint it was converted from.

use std::path::{Path, PathBuf};

use astronomical_model_serving::{
    QuantizedExpertLayerPlan, Qwen3_5ArtifactValidator, ValidatedQwen3_5Artifact,
    build_quantized_expert_page_manifest_from_plan, expert_layer_plans_for_validated_artifact,
    load_quantized_expert_page, mlx_ram_budget_model_geometry_from_validated_artifact,
};
use astronomical_runtime_integration::MlxRuntime;
use astronomical_test_harness::test_run::TestRunContext;
use serial_test::serial;

use super::large_sparse_moe_fixture::probe_runtime;

const MAXIMUM_OUTPUT_TOKENS: u32 = 512;

/// Routed pages chosen to cover adjacent runs (which the stock layout merges
/// into one read), scattered experts, and both ends of the expert range.
/// Ids derive from the expert count, sorted and unique, because the page
/// loader requires strictly increasing ids in range on any model size.
fn probe_expert_pages(expert_count: usize) -> Vec<Vec<usize>> {
    let scattered_expert_ids: std::collections::BTreeSet<usize> = [
        3,
        expert_count / 5,
        expert_count * 4 / 5,
        expert_count * 4 / 5 + 1,
        expert_count - 2,
        expert_count - 1,
    ]
    .into_iter()
    .filter(|&expert_id| expert_id < expert_count)
    .collect();
    vec![
        (0..expert_count.min(8)).collect(),
        scattered_expert_ids.into_iter().collect(),
        vec![expert_count / 2],
    ]
}

struct ValidatedPair {
    atlas_directory: PathBuf,
    atlas_artifact: ValidatedQwen3_5Artifact,
    stock_directory: PathBuf,
    stock_artifact: ValidatedQwen3_5Artifact,
}

fn validated_atlas_and_stock_artifacts() -> ValidatedPair {
    let test_run_configuration = TestRunContext::current().configuration;
    let atlas_directory = test_run_configuration
        .model_directory
        .expect("model_directory names the Atlas model");
    let stock_directory = test_run_configuration
        .reference_artifact_path
        .expect("reference_artifact_path names the stock checkpoint the Atlas came from");
    let validate = |model_directory: &Path| {
        Qwen3_5ArtifactValidator::new()
            .validate(model_directory, MAXIMUM_OUTPUT_TOKENS)
            .unwrap_or_else(|validation_error| {
                panic!(
                    "{} should validate: {validation_error}",
                    model_directory.display()
                )
            })
    };
    ValidatedPair {
        atlas_artifact: validate(&atlas_directory),
        stock_artifact: validate(&stock_directory),
        atlas_directory,
        stock_directory,
    }
}

#[test]
#[ignore = "loads routed pages and a complete layer from an installed Atlas model and its stock source"]
#[serial]
fn should_load_identical_expert_tensors_from_an_atlas_and_its_stock_source() {
    let validated_pair = validated_atlas_and_stock_artifacts();
    assert!(
        validated_pair.atlas_artifact.expert_atlas().is_some(),
        "the Atlas validates as an Atlas"
    );
    assert!(
        validated_pair.stock_artifact.expert_atlas().is_none(),
        "the source validates as a stock checkpoint"
    );
    assert_eq!(
        validated_pair.atlas_artifact.total_payload_bytes(),
        validated_pair.stock_artifact.total_payload_bytes(),
        "both layouts hold the same weight bytes"
    );
    assert_eq!(
        format!(
            "{:?}",
            mlx_ram_budget_model_geometry_from_validated_artifact(
                &validated_pair.atlas_artifact,
                &validated_pair.atlas_directory
            )
            .expect("Atlas geometry")
        ),
        format!(
            "{:?}",
            mlx_ram_budget_model_geometry_from_validated_artifact(
                &validated_pair.stock_artifact,
                &validated_pair.stock_directory
            )
            .expect("stock geometry")
        ),
        "memory budgeting sees the same model either way"
    );
    let atlas_layer_plans = expert_layer_plans_for_validated_artifact(
        &validated_pair.atlas_artifact,
        &validated_pair.atlas_directory,
    )
    .expect("Atlas plans build");
    let stock_layer_plans = expert_layer_plans_for_validated_artifact(
        &validated_pair.stock_artifact,
        &validated_pair.stock_directory,
    )
    .expect("stock plans build");
    assert_eq!(atlas_layer_plans.len(), stock_layer_plans.len());
    let runtime = probe_runtime();
    let last_layer_index = atlas_layer_plans.len() - 1;
    for layer_index in [0, last_layer_index / 2, last_layer_index] {
        for expert_ids in probe_expert_pages(atlas_layer_plans[layer_index].expert_capacity) {
            assert_same_page(
                &runtime,
                &atlas_layer_plans[layer_index],
                &stock_layer_plans[layer_index],
                &expert_ids,
            );
        }
    }
    let every_expert_id: Vec<usize> = (0..atlas_layer_plans[0].expert_capacity).collect();
    assert_same_page(
        &runtime,
        &atlas_layer_plans[0],
        &stock_layer_plans[0],
        &every_expert_id,
    );
}

/// Loads the page through both plans and requires every tensor to match in
/// name, shape and bytes.
fn assert_same_page(
    runtime: &MlxRuntime,
    atlas_layer_plan: &QuantizedExpertLayerPlan,
    stock_layer_plan: &QuantizedExpertLayerPlan,
    expert_ids: &[usize],
) {
    let load_page = |layer_plan: &QuantizedExpertLayerPlan| {
        let page_manifest = build_quantized_expert_page_manifest_from_plan(layer_plan, expert_ids)
            .expect("the page manifest builds");
        load_quantized_expert_page(runtime, &page_manifest, None).expect("the page loads")
    };
    let atlas_tensors = load_page(atlas_layer_plan);
    let stock_tensors = load_page(stock_layer_plan);
    assert_eq!(
        atlas_tensors.len(),
        stock_tensors.len(),
        "{}: both layouts present the same tensors",
        atlas_layer_plan.layer_prefix
    );
    for (tensor_name, stock_tensor) in &stock_tensors {
        let atlas_tensor = atlas_tensors.get(tensor_name).unwrap_or_else(|| {
            panic!(
                "{} {tensor_name} is missing from the Atlas page",
                atlas_layer_plan.layer_prefix
            )
        });
        assert_eq!(
            atlas_tensor.shape(),
            stock_tensor.shape(),
            "{} {tensor_name} shape",
            atlas_layer_plan.layer_prefix
        );
        let tensors_are_equal = runtime
            .array_equal(atlas_tensor, stock_tensor, false)
            .expect("array_equal builds");
        assert_eq!(
            tensors_are_equal
                .to_vec_bool()
                .expect("the comparison evaluates"),
            vec![true],
            "{} {tensor_name} bytes differ for experts {expert_ids:?}",
            atlas_layer_plan.layer_prefix
        );
    }
}
