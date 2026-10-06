import Foundation;
import ModelServing;
import IpcProtocol;
import Testing;
import JourneyCategories;

/**
 * Synthetic bounded shard-index fixtures, port of
 * crates/model-serving/tests/qwen3_5_hermetic/artifact_test_support.rs.
 */
enum Qwen3_5ArtifactTestSupport {

    static let FROZEN_LANGUAGE_PAYLOAD_BYTES: UInt64 = 22_164_699_392;
    private static let FROZEN_TOTAL_PARAMETERS: UInt64 = 34_660_608_768;
    static let LANGUAGE_SHARD_FILE_NAMES: Array<String> = [
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
    ];
    private static let VISION_SIDECAR_FILE_NAME: String = "vision/weights.safetensors";

    static func frozenTestIndexBytes() throws -> Array<UInt8> {
        let languageTensorProfiles: Array<TensorProfile> = try expectedLanguageTensorProfiles();
        return try frozenTestIndexBytesWithOptionalLanguageTensorReplacement(
            replacementTensor: nil,
            languageTensorProfiles: languageTensorProfiles);
    }

    static func frozenTestIndexBytesWithLanguageTensorReplacement(
        replacementTensorIndex: Int, replacementTensorName: String,
        languageTensorProfiles: Array<TensorProfile>) throws -> Array<UInt8> {
        return try frozenTestIndexBytesWithOptionalLanguageTensorReplacement(
            replacementTensor: (replacementTensorIndex, replacementTensorName),
            languageTensorProfiles: languageTensorProfiles);
    }

    /// The frozen Ornith-1.0-shaped MoE config drives the same language
    /// tensor-profile generator the engine uses.
    static func expectedLanguageTensorProfiles() throws -> Array<TensorProfile> {
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes());
        return Qwen3_5TensorSpec.qwen3_5LanguageTensorProfiles(qwen3_5Config: ornithConfig);
    }

    private static func frozenTestIndexBytesWithOptionalLanguageTensorReplacement(
        replacementTensor: (index: Int, name: String)?,
        languageTensorProfiles: Array<TensorProfile>) throws -> Array<UInt8> {
        var weightMapEntries: Array<(tensorName: String, shardFileName: String)> = Array();
        for (languageTensorIndex, tensorProfile): (Int, TensorProfile) in languageTensorProfiles.enumerated() {
            let tensorName: String;
            if replacementTensor?.index == languageTensorIndex,
                let replacementName: String = replacementTensor?.name {
                tensorName = replacementName;
            } else {
                tensorName = tensorProfile.name;
            }
            weightMapEntries.append((
                tensorName,
                LANGUAGE_SHARD_FILE_NAMES[languageTensorIndex % LANGUAGE_SHARD_FILE_NAMES.count]));
        }
        for visionTensorIndex in 0..<333 {
            weightMapEntries.append(
                ("vision_tower.synthetic.\(visionTensorIndex).weight", VISION_SIDECAR_FILE_NAME));
        }
        weightMapEntries.sort { (leftEntry: (tensorName: String, shardFileName: String), rightEntry: (tensorName: String, shardFileName: String)) -> Bool in
            return Array(leftEntry.tensorName.utf8).lexicographicallyPrecedes(Array(rightEntry.tensorName.utf8));
        };
        var weightMapObject: JsonWireObject = JsonWireObject(entries: Array());
        for weightEntry: (tensorName: String, shardFileName: String) in weightMapEntries {
            weightMapObject.appendEntry(key: weightEntry.tensorName, value: .string(weightEntry.shardFileName));
        }
        var metadataObject: JsonWireObject = JsonWireObject(entries: Array());
        metadataObject.appendEntry(key: "total_size", value: .unsignedInteger(FROZEN_LANGUAGE_PAYLOAD_BYTES));
        metadataObject.appendEntry(key: "total_parameters", value: .unsignedInteger(FROZEN_TOTAL_PARAMETERS));
        var indexObject: JsonWireObject = JsonWireObject(entries: Array());
        indexObject.appendEntry(key: "metadata", value: .object(metadataObject));
        indexObject.appendEntry(key: "weight_map", value: .object(weightMapObject));
        return Array(try JsonWireValue.object(indexObject).serializedText.utf8);
    }
}

/// Ported from crates/model-serving/tests/qwen3_5_hermetic/artifact.rs.
@Suite(.tags(.hermeticJourney))
final class Qwen3_5ArtifactShardIndexTests {

    @Test
    func should_exclude_a_nested_vision_sidecar_from_the_executable_model_shard_inventory() throws {
        let languageTensorProfiles: Array<TensorProfile> = try Qwen3_5ArtifactTestSupport.expectedLanguageTensorProfiles();
        let indexBytes: Array<UInt8> = try Qwen3_5ArtifactTestSupport.frozenTestIndexBytes();
        let shardIndex: Qwen3_5ShardIndex = try Qwen3_5ShardIndex.fromJsonBytes(
            indexBytes: indexBytes, languageTensorProfiles: languageTensorProfiles);
        #expect(
            shardIndex.totalPayloadBytes()
                == Qwen3_5ArtifactTestSupport.FROZEN_LANGUAGE_PAYLOAD_BYTES);
        #expect(shardIndex.tensorCount() == 1_757);
        #expect(shardIndex.languageTensorCount() == 1_757);
        #expect(
            shardIndex.modelShardFileNames()
                == Qwen3_5ArtifactTestSupport.LANGUAGE_SHARD_FILE_NAMES);
    }

    @Test
    func should_classify_a_root_vision_only_file_by_tensor_role_instead_of_filename() throws {
        let languageTensorProfiles: Array<TensorProfile> = try Qwen3_5ArtifactTestSupport.expectedLanguageTensorProfiles();
        let visionOnlyModelShardFileName: String = "model-vision-only.safetensors";
        let indexDocument: JsonWireValue = try Qwen3_5MoeConfigFixtures.wireValue(
            String(decoding: try Qwen3_5ArtifactTestSupport.frozenTestIndexBytes(), as: UTF8.self));
        guard case .object(let documentObject) = indexDocument else {
            Issue.record("the frozen synthetic shard index should be an object");
            return;
        }
        let weightMapObject: JsonWireObject = try documentObject.decodeObject(fieldName: "weight_map");
        var reclassifiedWeightMap: JsonWireObject = JsonWireObject(entries: Array());
        for entry: (key: String, value: JsonWireValue) in weightMapObject.entries {
            if entry.key.hasPrefix("vision_tower.") {
                reclassifiedWeightMap.appendEntry(key: entry.key, value: .string(visionOnlyModelShardFileName));
            } else {
                reclassifiedWeightMap.appendEntry(key: entry.key, value: entry.value);
            }
        }
        let indexBytes: Array<UInt8> = try Qwen3_5MoeConfigFixtures.serializedBytes(
            indexDocument.settingObjectKey(path: ["weight_map"], newValue: .object(reclassifiedWeightMap)));
        let shardIndex: Qwen3_5ShardIndex = try Qwen3_5ShardIndex.fromJsonBytes(
            indexBytes: indexBytes, languageTensorProfiles: languageTensorProfiles);
        #expect(
            shardIndex.modelShardFileNames().contains(visionOnlyModelShardFileName) == false);
        #expect(shardIndex.visionSidecarFileNames() == [visionOnlyModelShardFileName]);
    }

    @Test
    func should_reject_an_ornith_shard_index_with_an_unexpected_executable_language_tensor_name() throws {
        let languageTensorProfiles: Array<TensorProfile> = try Qwen3_5ArtifactTestSupport.expectedLanguageTensorProfiles();
        let unexpectedTensorName: String = "language_model.model.layers.0.linear_attn.in_proj_qkvz.weight";
        let indexBytes: Array<UInt8> = try Qwen3_5ArtifactTestSupport
            .frozenTestIndexBytesWithLanguageTensorReplacement(
                replacementTensorIndex: 0,
                replacementTensorName: unexpectedTensorName,
                languageTensorProfiles: languageTensorProfiles);
        do {
            _ = try Qwen3_5ShardIndex.fromJsonBytes(
                indexBytes: indexBytes, languageTensorProfiles: languageTensorProfiles);
            Issue.record("an unexpected executable language tensor must fail index validation");
        } catch let artifactError as Qwen3_5ArtifactError {
            #expect(
                artifactError == Qwen3_5ArtifactError.unexpectedLanguageTensor(
                    tensorName: unexpectedTensorName));
        }
    }
}
