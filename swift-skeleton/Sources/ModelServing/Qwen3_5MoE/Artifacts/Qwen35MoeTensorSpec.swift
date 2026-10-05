import Foundation;

/// Mixture-of-experts feed-forward tensor-profile helpers, port of
/// crates/model-serving/src/qwen3_5_moe/artifacts/tensor_spec.rs.
enum Qwen3_5MoeTensorSpec {

    static func appendQwen3_5MoeFeedForwardTensorProfiles(
        tensorProfiles: inout Array<TensorProfile>,
        decoderLayerPrefix: String,
        hiddenSize: Int,
        qwen3_5Config: Qwen3_5Config) {
        let mixtureOfExpertsPrefix: String = "\(decoderLayerPrefix).mlp";
        let routerModuleName: String = "\(mixtureOfExpertsPrefix).gate";
        let routerQuantizationProfile: OptiQQuantizationProfile = qwen3_5Config.quantizationProfile(forModule: routerModuleName);
        if routerQuantizationProfile.isUnquantized() {
            tensorProfiles.append(Qwen3_5TensorSpec.qwen3_5TensorProfile(
                tensorName: "\(routerModuleName).weight",
                tensorDtype: .modelFloat,
                tensorShape: [Int(qwen3_5Config.expertCount()), hiddenSize]));
        } else {
            Qwen3_5TensorSpec.appendQwen3_5QuantizedAffineTensorProfiles(
                tensorProfiles: &tensorProfiles,
                tensorPrefix: routerModuleName,
                leadingDimensions: [Int(qwen3_5Config.expertCount())],
                inputDimension: hiddenSize,
                quantizationProfile: routerQuantizationProfile);
        }
        let switchMlpProjectionShapes: Array<(projectionName: String, outputDimension: Int, inputDimension: Int)> = [
            ("gate_proj", Int(qwen3_5Config.expertIntermediateSize()), hiddenSize),
            ("up_proj", Int(qwen3_5Config.expertIntermediateSize()), hiddenSize),
            ("down_proj", hiddenSize, Int(qwen3_5Config.expertIntermediateSize())),
        ];
        for switchMlpProjection: (projectionName: String, outputDimension: Int, inputDimension: Int) in switchMlpProjectionShapes {
            let projectionModuleName: String = "\(mixtureOfExpertsPrefix).switch_mlp.\(switchMlpProjection.projectionName)";
            Qwen3_5TensorSpec.appendQwen3_5QuantizedAffineTensorProfiles(
                tensorProfiles: &tensorProfiles,
                tensorPrefix: projectionModuleName,
                leadingDimensions: [Int(qwen3_5Config.expertCount()), switchMlpProjection.outputDimension],
                inputDimension: switchMlpProjection.inputDimension,
                quantizationProfile: qwen3_5Config.quantizationProfile(forModule: projectionModuleName));
        }
        let sharedExpertProjectionShapes: Array<(projectionName: String, outputDimension: Int, inputDimension: Int)> = [
            ("gate_proj", Int(qwen3_5Config.sharedExpertIntermediateSize()), hiddenSize),
            ("up_proj", Int(qwen3_5Config.sharedExpertIntermediateSize()), hiddenSize),
            ("down_proj", hiddenSize, Int(qwen3_5Config.sharedExpertIntermediateSize())),
        ];
        for sharedExpertProjection: (projectionName: String, outputDimension: Int, inputDimension: Int) in sharedExpertProjectionShapes {
            let projectionModuleName: String = "\(mixtureOfExpertsPrefix).shared_expert.\(sharedExpertProjection.projectionName)";
            Qwen3_5TensorSpec.appendQwen3_5QuantizedAffineTensorProfiles(
                tensorProfiles: &tensorProfiles,
                tensorPrefix: projectionModuleName,
                leadingDimensions: [sharedExpertProjection.outputDimension],
                inputDimension: sharedExpertProjection.inputDimension,
                quantizationProfile: qwen3_5Config.quantizationProfile(forModule: projectionModuleName));
        }
        let sharedExpertGateModuleName: String = "\(mixtureOfExpertsPrefix).shared_expert_gate";
        Qwen3_5TensorSpec.appendQwen3_5QuantizedAffineTensorProfiles(
            tensorProfiles: &tensorProfiles,
            tensorPrefix: sharedExpertGateModuleName,
            leadingDimensions: [1],
            inputDimension: hiddenSize,
            quantizationProfile: qwen3_5Config.quantizationProfile(forModule: sharedExpertGateModuleName));
    }

    static func isSparseSelectedExpertTensorName(tensorName: String) -> Bool {
        return tensorName.contains(".mlp.switch_mlp.");
    }
}
