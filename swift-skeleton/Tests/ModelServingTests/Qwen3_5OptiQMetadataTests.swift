import Foundation;
import ModelServing;
import IpcProtocol;
import Testing;
import JourneyCategories;

/// Ported from crates/model-serving/tests/qwen3_5_hermetic/config/optiq_metadata.rs.
@Suite(.tags(.hermeticJourney))
final class Qwen3_5OptiQMetadataTests {

    @Test
    func should_accept_two_bit_optiq_metadata_supported_by_mlx_affine_quantization() throws {
        let optiQMetadataBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(
            try Qwen3_5MoeConfigFixtures.wireValue("""
        {
            "method": "static_mixed_precision",
            "base_model": "example/Qwen3.5-122B-A10B-bf16",
            "reference": "structural_rules",
            "target_bpw": 2.5,
            "achieved_bpw": 2.5,
            "n_high_bits": 0,
            "n_low_bits": 1,
            "threshold": 0.0,
            "per_layer": {
                "language_model.model.layers.5.mlp.switch_mlp.gate_proj": {
                    "bits": 2,
                    "group_size": 64
                }
            }
        }
        """));
        let optiQMetadata: OptiQMetadata = try OptiQMetadata.fromJsonBytes(
            metadataBytes: optiQMetadataBytes);
        #expect(optiQMetadata.measuredModuleCount() == 1);
    }

    @Test
    func should_require_the_optiq_metadata_bit_map_to_match_the_config() throws {
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: Qwen3_5MoeConfigFixtures.frozenOrnith10OptiQConfigBytes());
        let optiQMetadata: OptiQMetadata = try OptiQMetadata.fromJsonBytes(
            metadataBytes: Qwen3_5MoeConfigFixtures.frozenOptiQMetadataBytes());
        #expect(optiQMetadata.measuredModuleCount() == 510);
        try optiQMetadata.validateAgainstConfig(qwen3_5Config: ornithConfig);
    }

    @Test
    func should_accept_measured_optiq_profiles_that_are_a_strict_subset_of_the_config() throws {
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: Qwen3_5MoeConfigFixtures.frozenOrnith10OptiQConfigBytes());
        let metadataDocument: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOptiQMetadataBytes(), as: UTF8.self));
        let measuredModuleProfiles: JsonWireObject = try JsonWireValue.extractObject(
            metadataDocument.objectValue(forKey: "per_layer") ?? .null);
        var retainedProfiles: JsonWireObject = JsonWireObject(entries: Array());
        for entry: (key: String, value: JsonWireValue) in measuredModuleProfiles.entries {
            if entry.key.contains(".mlp.switch_mlp.") {
                continue;
            }
            retainedProfiles.appendEntry(key: entry.key, value: entry.value);
        }
        let subsetMetadataBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(
            metadataDocument.settingObjectKey(path: ["per_layer"], newValue: .object(retainedProfiles)));
        let optiQMetadata: OptiQMetadata = try OptiQMetadata.fromJsonBytes(
            metadataBytes: subsetMetadataBytes);
        #expect(optiQMetadata.measuredModuleCount() == 390);
        try optiQMetadata.validateAgainstConfig(qwen3_5Config: ornithConfig);
    }

    @Test
    func should_accept_optional_output_head_measurement_in_optiq_metadata() throws {
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: Qwen3_5MoeConfigFixtures.frozenOrnith10OptiQConfigBytes());
        let metadataDocument: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOptiQMetadataBytes(), as: UTF8.self));
        let outputHeadMetadataBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(
            metadataDocument.settingObjectKey(
                path: ["per_layer", "language_model.lm_head"],
                newValue: try Qwen3_5MoeConfigFixtures.wireValue(#"{"bits": 8, "group_size": 64}"#)));
        let optiQMetadata: OptiQMetadata = try OptiQMetadata.fromJsonBytes(
            metadataBytes: outputHeadMetadataBytes);
        try optiQMetadata.validateAgainstConfig(qwen3_5Config: ornithConfig);
    }

    @Test
    func should_compare_supported_optiq_metadata_group_sizes_with_the_model_config() throws {
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: Qwen3_5MoeConfigFixtures.frozenOrnith10OptiQConfigBytes());
        let metadataDocument: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: Qwen3_5MoeConfigFixtures.frozenOptiQMetadataBytes(), as: UTF8.self));
        let measuredModuleProfiles: JsonWireObject = try JsonWireValue.extractObject(
            metadataDocument.objectValue(forKey: "per_layer") ?? .null);
        guard let firstMeasuredEntry: (key: String, value: JsonWireValue) = measuredModuleProfiles.entries.first else {
            Issue.record("the frozen OptiQ metadata should measure at least one module");
            return;
        }
        let modifiedGroupSizeBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(
            metadataDocument.settingObjectKey(
                path: ["per_layer", firstMeasuredEntry.key, "group_size"],
                newValue: .unsignedInteger(32)));
        let optiQMetadata: OptiQMetadata = try OptiQMetadata.fromJsonBytes(
            metadataBytes: modifiedGroupSizeBytes);
        do {
            try optiQMetadata.validateAgainstConfig(qwen3_5Config: ornithConfig);
            Issue.record("a metadata group size that differs from config should fail validation");
        } catch let validationError as OptiQMetadataError {
            guard case .configGroupSizeMismatch(let moduleName, let configGroupSize, let metadataGroupSize) = validationError else {
                Issue.record("expected ConfigGroupSizeMismatch, got \(validationError)");
                return;
            }
            #expect(moduleName == firstMeasuredEntry.key);
            #expect(configGroupSize == 64);
            #expect(metadataGroupSize == 32);
        }
    }

    @Test
    func should_accept_provenance_only_optiq_metadata_without_a_measured_bit_map() throws {
        // Expert-compressed variants (for example REAP expert pruning) publish a
        // provenance document shaped around the compression run instead of a
        // sensitivity bit-map. It makes no per-module quantization claims, so it
        // must parse with zero measured modules and never block the load.
        let optiQMetadataBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(
            try Qwen3_5MoeConfigFixtures.wireValue("""
        {
            "expert_pruning": {
                "method": "reap",
                "parent_model": "example/example-parent",
                "num_experts_before": 256,
                "top_k": 8,
                "uniform_retention": true,
                "retention_target": 0.5
            }
        }
        """));
        let optiQMetadata: OptiQMetadata = try OptiQMetadata.fromJsonBytes(
            metadataBytes: optiQMetadataBytes);
        #expect(optiQMetadata.measuredModuleCount() == 0);
    }

    @Test
    func should_reject_measured_optiq_metadata_with_unknown_fields() throws {
        // Presence of `per_layer` keeps the exact strict contract: unknown fields
        // stay rejected so a measured document cannot smuggle undeclared content.
        let optiQMetadataBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(
            try Qwen3_5MoeConfigFixtures.wireValue("""
        {
            "method": "static_mixed_precision",
            "base_model": "example/example-parent",
            "reference": "structural_rules",
            "target_bpw": 4.0,
            "achieved_bpw": 4.0,
            "n_high_bits": 0,
            "n_low_bits": 1,
            "threshold": 0.0,
            "per_layer": {
                "language_model.model.layers.5.mlp.switch_mlp.gate_proj": {
                    "bits": 4,
                    "group_size": 64
                }
            },
            "unexpected_field": true
        }
        """));
        do {
            _ = try OptiQMetadata.fromJsonBytes(metadataBytes: optiQMetadataBytes);
            Issue.record("measured OptiQ metadata with unknown fields should stay rejected");
        } catch let metadataError as OptiQMetadataError {
            guard case .deserializeMetadata = metadataError else {
                Issue.record("expected DeserializeMetadata, got \(metadataError)");
                return;
            }
        }
    }
}
