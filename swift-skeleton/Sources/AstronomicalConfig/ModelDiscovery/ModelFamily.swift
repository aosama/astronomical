import Foundation;

/**
 * The model families discovery recognizes, ordered exactly like the Rust
 * `from_model_type` dispatch so the first matching verifier wins.
 */
internal enum ModelFamily: String, Equatable, Sendable {
    case qwen35 = "qwen3_5";
    case qwen4Exp = "qwen4_exp";
    case laguna = "laguna";
    case deepseekV4 = "deepseek_v4";
    case k2HorizonMova = "k2_horizon_mova";
    case flux2Klein = "flux2_klein";
    case qwenImage21 = "qwen_image_21";
    case modernbert = "modernbert";

    internal static func fromModelType(_ modelType: String?) -> ModelFamily? {
        if Qwen35.recognizesModelType(modelType) {
            return ModelFamily.qwen35;
        }
        if Qwen4Exp.recognizesModelType(modelType) {
            return ModelFamily.qwen4Exp;
        }
        if Laguna.recognizesModelType(modelType) {
            return ModelFamily.laguna;
        }
        if DeepseekV4.recognizesModelType(modelType) {
            return ModelFamily.deepseekV4;
        }
        if K2HorizonMova.recognizesModelType(modelType) {
            return ModelFamily.k2HorizonMova;
        }
        if Modernbert.recognizesModelType(modelType) {
            return ModelFamily.modernbert;
        }
        return nil;
    }
}
