mod artifact_support;
mod artifact_text_validation;
mod artifact_validation;
mod canonical_identity;
mod compressed_artifact_support;
mod compressed_schema;
mod compressed_storage;
mod compressed_storage_boundaries;
mod configuration;
mod direct_affine_artifact_validation;
mod exact_storage_edge_cases;
mod index_total_size;
mod kv_cache_metadata;
mod optional_artifacts;
mod rope;
mod storage;
mod strict_configuration;
mod support;
mod tensor_names;
mod text_artifacts;
mod text_generation;
mod text_output_parser;
mod text_support;

#[test]
fn should_keep_contract_only_laguna_unavailable_for_execution() {
    assert_eq!(
        astronomical_model_serving::laguna_unavailable_reason(),
        "Laguna model execution is not implemented in this build"
    );
}
