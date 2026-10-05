import Foundation;

/**
 * Family-owned shallow classification rules for DeepSeek-V4 artifacts,
 * porting crates/config/src/model_discovery/deepseek_v4.rs. The caseless
 * enum is the Swift equivalent of the Rust module of free functions. The
 * family is only recognized; execution support is claimed elsewhere.
 */
internal enum DeepseekV4 {

    /** Recognizes the DeepSeek-V4 family marker without claiming execution support. */
    internal static func recognizesModelType(_ modelType: String?) -> Bool {
        guard let presentModelType: String = modelType else {
            return false;
        }
        return presentModelType == "deepseek_v4";
    }
}
