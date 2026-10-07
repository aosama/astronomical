import Foundation;
import ModelServing;
import Testing;
import JourneyCategories;

/**
 * Behavioral journeys for the generated Qwen3.5-MoE language tensor profiles,
 * twin-porting crates/model-serving/tests/qwen3_5_moe_hermetic/tensor_spec.rs.
 * The profile counts and shapes are structural consequences of the accepted
 * config, not golden-master constants tied to one packaging variant.
 */
@Suite(.tags(.hermeticJourney))
final class Qwen3_5MoeTensorSpecTests {

    @Test
    func should_generate_the_complete_mixed_precision_optiq_language_tensor_profile() throws {
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: Qwen3_5MoeConfigFixtures.frozenOrnith10OptiQConfigBytes());

        let tensorProfiles: Array<TensorProfile> = Qwen3_5TensorSpec.qwen3_5LanguageTensorProfiles(
            qwen3_5Config: ornithConfig);
        let profileByName: Dictionary<String, TensorProfile> = tensorProfileByNameDictionary(tensorProfiles);

        #expect(tensorProfiles.count == 1_757);
        #expect(profileByName.count == 1_757);
        assertTensor(
            profileByName, "language_model.model.embed_tokens.weight", .uint32, [248_320, 512]);
        assertTensor(
            profileByName, "language_model.model.embed_tokens.scales",
            .affineQuantizationFloat, [248_320, 32]);
        assertTensor(
            profileByName, "language_model.model.layers.0.linear_attn.conv1d.weight",
            .modelFloat, [8_192, 4, 1]);
        assertTensor(
            profileByName, "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
            .uint32, [8_192, 256]);
        assertTensor(
            profileByName, "language_model.model.layers.0.linear_attn.in_proj_z.scales",
            .affineQuantizationFloat, [4_096, 32]);
        assertTensor(
            profileByName, "language_model.model.layers.0.linear_attn.in_proj_b.biases",
            .affineQuantizationFloat, [32, 32]);
        assertTensor(
            profileByName, "language_model.model.layers.0.linear_attn.A_log", .modelFloat, [32]);
        assertTensor(
            profileByName, "language_model.model.layers.0.linear_attn.dt_bias", .modelFloat, [32]);
        assertTensor(
            profileByName, "language_model.model.layers.0.linear_attn.norm.weight",
            .modelFloat, [128]);
        assertTensor(
            profileByName, "language_model.model.layers.0.mlp.gate.weight", .uint32, [256, 256]);
        assertTensor(
            profileByName, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
            .uint32, [256, 512, 256]);
        assertTensor(
            profileByName, "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales",
            .affineQuantizationFloat, [256, 2_048, 8]);
        assertTensor(
            profileByName, "language_model.model.layers.0.mlp.shared_expert.down_proj.weight",
            .uint32, [2_048, 64]);
        assertTensor(
            profileByName, "language_model.model.layers.0.mlp.shared_expert_gate.weight",
            .uint32, [1, 256]);
        assertTensor(
            profileByName, "language_model.model.layers.3.self_attn.q_proj.weight",
            .uint32, [8_192, 256]);
        assertTensor(
            profileByName, "language_model.model.layers.39.self_attn.q_proj.weight",
            .uint32, [8_192, 512]);
        assertTensor(
            profileByName, "language_model.model.layers.3.self_attn.o_proj.scales",
            .affineQuantizationFloat, [2_048, 64]);
        assertTensor(
            profileByName, "language_model.model.layers.3.self_attn.q_norm.weight",
            .modelFloat, [256]);
        assertTensor(
            profileByName, "language_model.model.layers.39.self_attn.k_proj.biases",
            .affineQuantizationFloat, [512, 32]);
        assertTensor(
            profileByName, "language_model.model.norm.weight", .modelFloat, [2_048]);
        assertTensor(
            profileByName, "language_model.lm_head.biases",
            .affineQuantizationFloat, [248_320, 32]);
        #expect(countTensorsInLayer(tensorProfiles, decoderLayerIndex: 0) == 45);
        #expect(countTensorsInLayer(tensorProfiles, decoderLayerIndex: 3) == 40);
        let hasMergedProjectionNames: Bool = profileByName.keys.contains { tensorName in
            tensorName.contains("in_proj_qkvz") || tensorName.contains("in_proj_ba")
        };
        #expect(hasMergedProjectionNames == false);
        let hasVisionTowerNames: Bool = profileByName.keys.contains { tensorName in
            tensorName.hasPrefix("vision_tower.")
        };
        #expect(hasVisionTowerNames == false);
    }

    @Test
    func should_exclude_sparse_expert_tensors_from_every_resident_profile() throws {
        let ornithConfig: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: Qwen3_5MoeConfigFixtures.frozenOrnith10OptiQConfigBytes());

        let residentTensorProfiles: Array<TensorProfile> = Qwen3_5TensorSpec
            .qwen3_5ResidentLanguageTensorProfiles(qwen3_5Config: ornithConfig);
        let residentProfileByName: Dictionary<String, TensorProfile> =
            tensorProfileByNameDictionary(residentTensorProfiles);

        #expect(residentTensorProfiles.count == 1_397);
        #expect(countTensorsInLayer(residentTensorProfiles, decoderLayerIndex: 0) == 36);
        #expect(countTensorsInLayer(residentTensorProfiles, decoderLayerIndex: 3) == 31);
        let hasSparseExpertNames: Bool = residentProfileByName.keys.contains { tensorName in
            tensorName.contains(".mlp.switch_mlp.")
        };
        #expect(
            hasSparseExpertNames == false,
            "resident profile must not bind sparse selected experts");
        assertTensor(
            residentProfileByName, "language_model.model.layers.0.mlp.gate.weight",
            .uint32, [256, 256]);
        assertTensor(
            residentProfileByName, "language_model.model.layers.0.mlp.shared_expert.down_proj.weight",
            .uint32, [2_048, 64]);
        assertTensor(
            residentProfileByName, "language_model.model.layers.0.mlp.shared_expert_gate.weight",
            .uint32, [1, 256]);
    }

    private func tensorProfileByNameDictionary(
        _ tensorProfiles: Array<TensorProfile>
    ) -> Dictionary<String, TensorProfile> {
        var profileByName: Dictionary<String, TensorProfile> = [:];
        for tensorProfile: TensorProfile in tensorProfiles {
            profileByName[tensorProfile.name] = tensorProfile;
        }
        return profileByName;
    }

    private func assertTensor(
        _ profileByName: Dictionary<String, TensorProfile>,
        _ tensorName: String,
        _ expectedDtype: TensorDtype,
        _ expectedShape: Array<Int>
    ) {
        guard let tensorProfile: TensorProfile = profileByName[tensorName] else {
            Issue.record("expected tensor profile \(tensorName)");
            return;
        }
        #expect(tensorProfile.dtype == expectedDtype, "dtype for \(tensorName)");
        #expect(tensorProfile.shape == expectedShape, "shape for \(tensorName)");
    }

    private func countTensorsInLayer(
        _ tensorProfiles: Array<TensorProfile>, decoderLayerIndex: Int
    ) -> Int {
        let layerPrefix: String = "language_model.model.layers.\(decoderLayerIndex).";
        var tensorCount: Int = 0;
        for tensorProfile: TensorProfile in tensorProfiles
        where tensorProfile.name.hasPrefix(layerPrefix) {
            tensorCount += 1;
        }
        return tensorCount;
    }
}
