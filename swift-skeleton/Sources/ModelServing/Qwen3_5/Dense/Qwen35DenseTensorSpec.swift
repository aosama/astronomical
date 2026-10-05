import Foundation;

/// Dense SwiGLU MLP tensor-profile helpers, port of
/// crates/model-serving/src/qwen3_5/dense/tensor_spec.rs.
enum Qwen3_5DenseTensorSpec {

    static func appendQwen3_5DenseMlpTensorProfiles(
        tensorProfiles: inout Array<TensorProfile>,
        decoderLayerPrefix: String,
        hiddenSize: Int,
        qwen3_5Config: Qwen3_5Config) {
        let denseMlpPrefix: String = "\(decoderLayerPrefix).mlp";
        let denseProjectionShapes: Array<(projectionName: String, outputDimension: Int, inputDimension: Int)> = [
            ("gate_proj", Int(qwen3_5Config.denseIntermediateSize()), hiddenSize),
            ("up_proj", Int(qwen3_5Config.denseIntermediateSize()), hiddenSize),
            ("down_proj", hiddenSize, Int(qwen3_5Config.denseIntermediateSize())),
        ];
        for denseProjection: (projectionName: String, outputDimension: Int, inputDimension: Int) in denseProjectionShapes {
            let projectionModuleName: String = "\(denseMlpPrefix).\(denseProjection.projectionName)";
            Qwen3_5TensorSpec.appendQwen3_5QuantizedAffineTensorProfiles(
                tensorProfiles: &tensorProfiles,
                tensorPrefix: projectionModuleName,
                leadingDimensions: [denseProjection.outputDimension],
                inputDimension: denseProjection.inputDimension,
                quantizationProfile: qwen3_5Config.quantizationProfile(forModule: projectionModuleName));
        }
    }
}
