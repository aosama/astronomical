import Foundation;

/// Expected checkpoint tensor shapes for the Qwen3.5 vision graph. Port of
/// crates/model-serving/src/qwen3_5/vision/vision_tensor_spec.rs.
///
/// Linear weights follow `[output_features, input_features]`; model execution
/// transposes them before the MLX addmm fusion. Conv3d weights already follow
/// MLX's `[output_channels, kernel_depth, kernel_height, kernel_width,
/// input_channels]` layout. These profiles therefore document the boundary
/// between stored model tensors and the MLX operation contracts.
public enum Qwen35VisionTensorSpec {

    /// Generates config-derived Qwen3.5 vision tower tensor metadata.
    ///
    /// Stored floating tensors retain any MLX-supported model float dtype and
    /// are grouped into: transformer blocks (12 tensors each), patch embedding
    /// (weight + bias), positional embedding (weight), and merger (norm +
    /// linear_fc1 + linear_fc2).
    public static func visionTensorProfiles(
        visionConfig: Qwen3_5VisionConfig) -> Array<TensorProfile> {
        let hiddenSize: Int = Int(visionConfig.hiddenSize);
        let intermediateSize: Int = Int(visionConfig.intermediateSize);
        let depth: Int = Int(visionConfig.depth);
        let patchSize: Int = Int(visionConfig.patchSize);
        let temporalPatchSize: Int = Int(visionConfig.temporalPatchSize);
        let inChannels: Int = Int(visionConfig.inChannels);
        let spatialMergeSize: Int = Int(visionConfig.spatialMergeSize);
        let positionEmbeddingCount: Int = Int(visionConfig.positionEmbeddingCount);
        let outHiddenSize: Int = Int(visionConfig.outHiddenSize);

        let qkvDimension: Int = hiddenSize * 3;
        let mergerInputDimension: Int = hiddenSize * spatialMergeSize * spatialMergeSize;

        var tensorProfiles: Array<TensorProfile> = Array();

        // Patch embedding: [hidden_size, temporal_patch_size, patch_size,
        // patch_size, in_channels].
        //
        // MLX Conv3d consumes ODHWI weights, and the converted artifacts store
        // that order. Some published checkpoints keep the upstream PyTorch
        // Conv3d order [hidden_size, in_channels, temporal_patch_size,
        // patch_size, patch_size] instead — a pure axis permutation of the
        // same values — so the profile accepts it and the vision weights
        // loader normalizes it at load.
        tensorProfiles.append(TensorProfile(
            name: "vision_tower.patch_embed.proj.weight",
            dtype: .modelFloat,
            shape: [hiddenSize, temporalPatchSize, patchSize, patchSize, inChannels],
            equivalentPublishedShapes: [[hiddenSize, inChannels, temporalPatchSize, patchSize, patchSize]]));
        tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
            tensorName: "vision_tower.patch_embed.proj.bias",
            tensorDtype: .modelFloat, tensorShape: [hiddenSize]));

        // Positional embedding: [position_embedding_count, hidden_size]
        tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
            tensorName: "vision_tower.pos_embed.weight",
            tensorDtype: .modelFloat, tensorShape: [positionEmbeddingCount, hiddenSize]));

        // Transformer blocks
        for blockIndex: Int in 0..<depth {
            let blockPrefix: String = "vision_tower.blocks.\(blockIndex)";

            // Attention QKV: [3*hidden_size, hidden_size] + bias [3*hidden_size]
            tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
                tensorName: blockPrefix + ".attn.qkv.weight",
                tensorDtype: .modelFloat, tensorShape: [qkvDimension, hiddenSize]));
            tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
                tensorName: blockPrefix + ".attn.qkv.bias",
                tensorDtype: .modelFloat, tensorShape: [qkvDimension]));

            // Attention projection: [hidden_size, hidden_size] + bias [hidden_size]
            tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
                tensorName: blockPrefix + ".attn.proj.weight",
                tensorDtype: .modelFloat, tensorShape: [hiddenSize, hiddenSize]));
            tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
                tensorName: blockPrefix + ".attn.proj.bias",
                tensorDtype: .modelFloat, tensorShape: [hiddenSize]));

            // LayerNorm 1: [hidden_size] x2 (weight + bias)
            tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
                tensorName: blockPrefix + ".norm1.weight",
                tensorDtype: .modelFloat, tensorShape: [hiddenSize]));
            tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
                tensorName: blockPrefix + ".norm1.bias",
                tensorDtype: .modelFloat, tensorShape: [hiddenSize]));

            // MLP fc1: [intermediate_size, hidden_size] + bias [intermediate_size]
            tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
                tensorName: blockPrefix + ".mlp.linear_fc1.weight",
                tensorDtype: .modelFloat, tensorShape: [intermediateSize, hiddenSize]));
            tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
                tensorName: blockPrefix + ".mlp.linear_fc1.bias",
                tensorDtype: .modelFloat, tensorShape: [intermediateSize]));

            // MLP fc2: [hidden_size, intermediate_size] + bias [hidden_size]
            tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
                tensorName: blockPrefix + ".mlp.linear_fc2.weight",
                tensorDtype: .modelFloat, tensorShape: [hiddenSize, intermediateSize]));
            tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
                tensorName: blockPrefix + ".mlp.linear_fc2.bias",
                tensorDtype: .modelFloat, tensorShape: [hiddenSize]));

            // LayerNorm 2: [hidden_size] x2 (weight + bias)
            tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
                tensorName: blockPrefix + ".norm2.weight",
                tensorDtype: .modelFloat, tensorShape: [hiddenSize]));
            tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
                tensorName: blockPrefix + ".norm2.bias",
                tensorDtype: .modelFloat, tensorShape: [hiddenSize]));
        }

        // The 2x2 spatial merge concatenates four hidden rows, so both fc1 axes
        // are merger_input_dimension=4*hidden_size. fc2 projects that vector to
        // the text model's out_hidden_size.
        tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
            tensorName: "vision_tower.merger.norm.weight",
            tensorDtype: .modelFloat, tensorShape: [hiddenSize]));
        tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
            tensorName: "vision_tower.merger.norm.bias",
            tensorDtype: .modelFloat, tensorShape: [hiddenSize]));
        tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
            tensorName: "vision_tower.merger.linear_fc1.weight",
            tensorDtype: .modelFloat, tensorShape: [mergerInputDimension, mergerInputDimension]));
        tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
            tensorName: "vision_tower.merger.linear_fc1.bias",
            tensorDtype: .modelFloat, tensorShape: [mergerInputDimension]));
        tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
            tensorName: "vision_tower.merger.linear_fc2.weight",
            tensorDtype: .modelFloat, tensorShape: [outHiddenSize, mergerInputDimension]));
        tensorProfiles.append(Qwen35VisionTensorSpec.tensorProfile(
            tensorName: "vision_tower.merger.linear_fc2.bias",
            tensorDtype: .modelFloat, tensorShape: [outHiddenSize]));

        return tensorProfiles;
    }

    private static func tensorProfile(
        tensorName: String, tensorDtype: TensorDtype,
        tensorShape: Array<Int>) -> TensorProfile {
        return TensorProfile(
            name: tensorName, dtype: tensorDtype, shape: tensorShape,
            equivalentPublishedShapes: []);
    }
}
