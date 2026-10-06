import Foundation;

/** License vocabulary attached to discovered models, mirroring `ModelLicense`. */
public enum ModelLicense: String, Equatable, Sendable {
    case apache20 = "Apache-2.0";
    case qwenResearch = "qwen-research";

    /** SPDX-style identifier reported through configuration APIs. */
    public var spdxIdentifier: String {
        return self.rawValue;
    }
}
