import XCTest;
import ModelServing;

/// Behavioral journeys for the generated Qwen3.5 vision tensor profiles,
/// twin-porting
/// crates/model-serving/tests/qwen3_5_hermetic/vision_tensor_spec.rs. The
/// profile count and shapes are structural consequences of the accepted
/// config, not golden-master constants tied to one packaging variant.
final class Qwen35VisionTensorSpecTests: XCTestCase {

    func testShouldGenerateTheCompleteOrnithVisionTensorProfile() throws {
        let visionConfig: Qwen3_5VisionConfig = try Qwen3_5VisionConfig.fromJsonBytes(
            configBytes: Data(Qwen35VisionConfigFixtures.FROZEN_VISION_CONFIG_JSON.utf8));
        let visionTensorProfiles: Array<TensorProfile> = Qwen35VisionTensorSpec
            .visionTensorProfiles(visionConfig: visionConfig);

        // 27 blocks x 12 + 3 patch/pos + 6 merger
        XCTAssertEqual(
            visionTensorProfiles.count, 333,
            "vision tensor profile must have exactly 333 tensors");

        // Patch embed
        let patchWeight: TensorProfile = visionTensorProfiles[0];
        XCTAssertEqual(patchWeight.name, "vision_tower.patch_embed.proj.weight");
        XCTAssertEqual(patchWeight.shape, [1152, 2, 16, 16, 3]);
        XCTAssertEqual(
            patchWeight.equivalentPublishedShapes, [[1152, 3, 2, 16, 16]],
            "the upstream PyTorch Conv3d order must be accepted and normalized to ODHWI");

        let patchBias: TensorProfile = visionTensorProfiles[1];
        XCTAssertEqual(patchBias.name, "vision_tower.patch_embed.proj.bias");
        XCTAssertEqual(patchBias.shape, [1152]);
        XCTAssertTrue(patchBias.equivalentPublishedShapes.isEmpty);

        // Positional embedding
        let posEmbed: TensorProfile = visionTensorProfiles[2];
        XCTAssertEqual(posEmbed.name, "vision_tower.pos_embed.weight");
        XCTAssertEqual(posEmbed.shape, [2304, 1152]);

        // First block attention
        let block0QkvWeight: TensorProfile = visionTensorProfiles[3];
        XCTAssertEqual(block0QkvWeight.name, "vision_tower.blocks.0.attn.qkv.weight");
        XCTAssertEqual(block0QkvWeight.shape, [3456, 1152]);

        // Last block norm2 bias (block 26, 12th tensor = index 3 + 26*12 + 11)
        let block26Norm2Bias: TensorProfile = visionTensorProfiles[3 + 26 * 12 + 11];
        XCTAssertEqual(block26Norm2Bias.name, "vision_tower.blocks.26.norm2.bias");
        XCTAssertEqual(block26Norm2Bias.shape, [1152]);

        // Merger (last 6 tensors: indices 327..333)
        let mergerNormWeight: TensorProfile = visionTensorProfiles[327];
        XCTAssertEqual(mergerNormWeight.name, "vision_tower.merger.norm.weight");
        XCTAssertEqual(mergerNormWeight.shape, [1152]);

        let mergerFc1Weight: TensorProfile = visionTensorProfiles[329];
        XCTAssertEqual(mergerFc1Weight.name, "vision_tower.merger.linear_fc1.weight");
        XCTAssertEqual(mergerFc1Weight.shape, [4608, 4608]);

        let mergerFc2Weight: TensorProfile = visionTensorProfiles[331];
        XCTAssertEqual(mergerFc2Weight.name, "vision_tower.merger.linear_fc2.weight");
        XCTAssertEqual(mergerFc2Weight.shape, [2048, 4608]);

        let mergerFc2Bias: TensorProfile = visionTensorProfiles[332];
        XCTAssertEqual(mergerFc2Bias.name, "vision_tower.merger.linear_fc2.bias");
        XCTAssertEqual(mergerFc2Bias.shape, [2048]);

        // Every stored floating tensor must retain an MLX-supported model dtype.
        for tensorProfile: TensorProfile in visionTensorProfiles {
            XCTAssertEqual(
                tensorProfile.dtype, TensorDtype.modelFloat,
                "tensor \(tensorProfile.name) must retain a supported stored model float dtype");
        }
    }
}
