import Foundation;

/**
 * The model families discovery recognizes, ordered exactly like the Rust
 * `from_model_type` dispatch so the first matching verifier wins.
 */
internal enum ModelFamily: String, Equatable, Sendable {
    case qwen35 = "qwen3_5";
    case k2HorizonMova = "k2_horizon_mova";
    case flux2Klein = "flux2_klein";
    case qwenImage21 = "qwen_image_21";
    case modernbert = "modernbert";

    internal static func fromModelType(_ modelType: String?) -> ModelFamily? {
        if Qwen35.recognizesModelType(modelType) {
            return ModelFamily.qwen35;
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
