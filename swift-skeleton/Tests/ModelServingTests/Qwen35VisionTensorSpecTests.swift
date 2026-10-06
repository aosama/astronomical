import Foundation;
import ModelServing;
import Testing;
import JourneyCategories;

/**
 * Behavioral journeys for the generated Qwen3.5 vision tensor profiles,
 * twin-porting
 * crates/model-serving/tests/qwen3_5_hermetic/vision_tensor_spec.rs. The
 * profile count and shapes are structural consequences of the accepted
 * config, not golden-master constants tied to one packaging variant.
 */
@Suite(.tags(.hermeticJourney))
final class Qwen35VisionTensorSpecTests {

    @Test
    func should_generate_the_complete_ornith_vision_tensor_profile() throws {
        let visionConfig: Qwen3_5VisionConfig = try Qwen3_5VisionConfig.fromJsonBytes(
            configBytes: Data(Qwen35VisionConfigFixtures.FROZEN_VISION_CONFIG_JSON.utf8));
        let visionTensorProfiles: Array<TensorProfile> = Qwen35VisionTensorSpec
            .visionTensorProfiles(visionConfig: visionConfig);

        // 27 blocks x 12 + 3 patch/pos + 6 merger
        #expect(
            visionTensorProfiles.count == 333,
            "vision tensor profile must have exactly 333 tensors");

        // Patch embed
        let patchWeight: TensorProfile = visionTensorProfiles[0];
        #expect(patchWeight.name == "vision_tower.patch_embed.proj.weight");
        #expect(patchWeight.shape == [1152, 2, 16, 16, 3]);
        #expect(
            patchWeight.equivalentPublishedShapes == [[1152, 3, 2, 16, 16]],
            "the upstream PyTorch Conv3d order must be accepted and normalized to ODHWI");

        let patchBias: TensorProfile = visionTensorProfiles[1];
        #expect(patchBias.name == "vision_tower.patch_embed.proj.bias");
        #expect(patchBias.shape == [1152]);
        #expect(patchBias.equivalentPublishedShapes.isEmpty);

        // Positional embedding
        let posEmbed: TensorProfile = visionTensorProfiles[2];
        #expect(posEmbed.name == "vision_tower.pos_embed.weight");
        #expect(posEmbed.shape == [2304, 1152]);

        // First block attention
        let block0QkvWeight: TensorProfile = visionTensorProfiles[3];
        #expect(block0QkvWeight.name == "vision_tower.blocks.0.attn.qkv.weight");
        #expect(block0QkvWeight.shape == [3456, 1152]);

        // Last block norm2 bias (block 26, 12th tensor = index 3 + 26*12 + 11)
        let block26Norm2Bias: TensorProfile = visionTensorProfiles[3 + 26 * 12 + 11];
        #expect(block26Norm2Bias.name == "vision_tower.blocks.26.norm2.bias");
        #expect(block26Norm2Bias.shape == [1152]);

        // Merger (last 6 tensors: indices 327..333)
        let mergerNormWeight: TensorProfile = visionTensorProfiles[327];
        #expect(mergerNormWeight.name == "vision_tower.merger.norm.weight");
        #expect(mergerNormWeight.shape == [1152]);

        let mergerFc1Weight: TensorProfile = visionTensorProfiles[329];
        #expect(mergerFc1Weight.name == "vision_tower.merger.linear_fc1.weight");
        #expect(mergerFc1Weight.shape == [4608, 4608]);

        let mergerFc2Weight: TensorProfile = visionTensorProfiles[331];
        #expect(mergerFc2Weight.name == "vision_tower.merger.linear_fc2.weight");
        #expect(mergerFc2Weight.shape == [2048, 4608]);

        let mergerFc2Bias: TensorProfile = visionTensorProfiles[332];
        #expect(mergerFc2Bias.name == "vision_tower.merger.linear_fc2.bias");
        #expect(mergerFc2Bias.shape == [2048]);

        // Every stored floating tensor must retain an MLX-supported model dtype.
        for tensorProfile: TensorProfile in visionTensorProfiles {
            #expect(
                tensorProfile.dtype == TensorDtype.modelFloat,
                "tensor \(tensorProfile.name) must retain a supported stored model float dtype");
        }
    }
}
